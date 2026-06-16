"""Asynchronous tool executor with lease-based deduplication.

The ToolWorker polls the event store for pending tool invocations,
respects dependency chains and deadlines, executes the actual tool
logic, and commits the result back as a ``tool_result_received`` event.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from rfsn_agent.common import utc_now
from rfsn_agent.domain import HarnessSnapshot
from rfsn_agent.events import (
    ProposedEvent,
    ToolResultReceivedPayload,
)
from rfsn_agent.security import SecurityProfile
from rfsn_agent.store import SQLiteEventStore
from rfsn_agent.types import ContentHash, ToolInvocationId, TrajectoryId


class ToolExecutionError(RuntimeError):
    """Raised when a tool invocation fails at runtime."""


ToolFunc = Callable[[str, tuple[tuple[str, str], ...]], Awaitable[str]]


class ToolWorker:
    """Async worker that executes pending tool calls from the event store.

    Each invocation is protected by an in-memory lease so multiple
    parallel workers will not execute the same call twice.
    """

    def __init__(
        self,
        store: SQLiteEventStore,
        security_profile: SecurityProfile | None = None,
        tool_registry: dict[str, ToolFunc] | None = None,
    ) -> None:
        self.store = store
        self.security = security_profile
        self._claim_lock = asyncio.Lock()
        self._lease_owners: dict[str, asyncio.Task[Any]] = {}
        self._registry = dict(tool_registry or {})
        self._running = False
        self._task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    async def start(self, poll_interval_seconds: float = 1.0) -> None:
        """Start the background polling loop."""
        self._running = True
        self._task = asyncio.create_task(
            self._poll_loop(poll_interval_seconds)
        )

    async def stop(self) -> None:
        """Stop the background polling loop gracefully."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def process_pending(self, trajectory_id: str | None = None) -> int:
        """Process all currently pending tool invocations.

        If ``trajectory_id`` is None, all trajectories are scanned.
        Returns the number of results committed.
        """
        count = 0
        trajectories = [trajectory_id] if trajectory_id else self.store.list_trajectories()
        for tid in trajectories:
            count += await self._process_trajectory(tid)
        return count

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _poll_loop(self, interval: float) -> None:
        while self._running:
            try:
                await self.process_pending()
            except Exception:
                pass
            await asyncio.sleep(interval)

    async def _process_trajectory(self, trajectory_id: str) -> int:
        snapshot = self.store.get_latest_snapshot(trajectory_id)
        pending = self._find_pending_invocations(snapshot)
        if not pending:
            return 0

        # Process each pending invocation concurrently.
        results = await asyncio.gather(
            *[self._execute_one(tid, inv) for tid, inv in pending],
            return_exceptions=True,
        )

        # Re-read snapshot: another worker may have committed results
        # while we were executing tools.
        snapshot = self.store.get_latest_snapshot(trajectory_id)
        completed = {r.invocation_id for r in snapshot.tool_results}

        # Filter proposals whose invocation is still uncommitted.
        valid_proposals: list[ProposedEvent] = []
        for result in results:
            if isinstance(result, ProposedEvent):
                payload = result.payload
                if (
                    isinstance(payload, ToolResultReceivedPayload)
                    and payload.invocation_id not in completed
                ):
                    valid_proposals.append(result)
                    completed.add(payload.invocation_id)

        if valid_proposals:
            self._commit_results(
                trajectory_id,
                snapshot.sequence,
                snapshot.last_event_hash,
                valid_proposals,
            )

        # Release all claims we held for this batch.
        for _, inv_id in pending:
            async with self._claim_lock:
                self._lease_owners.pop(inv_id, None)

        return len(valid_proposals)

    def _find_pending_invocations(
        self, snapshot: HarnessSnapshot
    ) -> list[tuple[str, ToolInvocationId]]:
        """Return (trajectory_id, invocation_id) for invocations without results."""
        completed = {r.invocation_id for r in snapshot.tool_results}
        pending: list[tuple[str, ToolInvocationId]] = [
            (str(snapshot.trajectory_id), t.invocation_id)
            for t in snapshot.tool_invocations
            if t.invocation_id not in completed
        ]
        return pending

    async def _execute_one(
        self, trajectory_id: str, invocation_id: ToolInvocationId
    ) -> ProposedEvent | None:
        """Execute a single tool invocation under a lease."""
        async with self._claim_lock:
            if invocation_id in self._lease_owners:
                return None  # Another worker already holds the lease.
            current = asyncio.current_task()
            assert current is not None
            self._lease_owners[invocation_id] = current

        # Re-read snapshot to avoid stale state.
        snapshot = self.store.get_latest_snapshot(trajectory_id)
        invocation = next(
            (t for t in snapshot.tool_invocations if t.invocation_id == invocation_id),
            None,
        )
        if invocation is None:
            return None  # Gone (e.g. rolled back).

        # Deduplication: check again for a result.
        if any(r.invocation_id == invocation_id for r in snapshot.tool_results):
            return None

        # Security: tool must be in allow-list.
        if self.security is not None and not self.security.is_tool_allowed(invocation.tool_name):
            return self._make_result(
                invocation_id, "failure",
                f"Tool '{invocation.tool_name}' not allowed by security profile"
            )

        # Security: read_file paths must not match forbidden pattern.
        if invocation.tool_name == "read_file" and self.security is not None:
            args_dict = dict(invocation.arguments)
            path = args_dict.get("source_id") or args_dict.get("path", "")
            if not self.security.is_path_allowed(path):
                return self._make_result(
                    invocation_id, "failure",
                    f"Path '{path}' matches forbidden path pattern"
                )

        # Dependency check: only run if all dependencies succeeded.
        result_map = {r.invocation_id: r.status.value for r in snapshot.tool_results}
        for dep_id in invocation.dependency_ids:
            dep_status = result_map.get(dep_id)
            if dep_status != "success":
                # Dependencies not yet ready: skip without recording anything.
                return None

        # Deadline check.
        if invocation.deadline is not None and utc_now() > invocation.deadline:
            return self._make_result(
                invocation_id, "timeout", "deadline exceeded"
            )

        # Execute the tool.
        try:
            content = await self._run_tool(invocation.tool_name, invocation.arguments)
            return self._make_result(invocation_id, "success", content)
        except Exception as exc:
            return self._make_result(
                invocation_id, "failure", f"{type(exc).__name__}: {exc}"
            )

    async def _run_tool(
        self, tool_name: str, arguments: tuple[tuple[str, str], ...]
    ) -> str:
        """Dispatch to the registered tool function or built-in defaults."""
        if tool_name in self._registry:
            return await self._registry[tool_name](tool_name, arguments)

        if tool_name == "web_search":
            query = dict(arguments).get("query", "")
            return json.dumps({"query": query, "results": "placeholder"})

        if tool_name == "read_file":
            args_dict = dict(arguments)
            path = args_dict.get("source_id") or args_dict.get("path", "")
            target = Path(path)
            if not target.exists():
                raise ToolExecutionError(f"File not found: {path}")
            return target.read_text(encoding="utf-8")

        raise ToolExecutionError(f"Unknown tool: {tool_name}")

    @staticmethod
    def _make_result(
        invocation_id: ToolInvocationId, status: str, content: str
    ) -> ProposedEvent:
        return ProposedEvent(
            event_type="tool_result_received",
            payload=ToolResultReceivedPayload(
                invocation_id=invocation_id,
                status=status,
                content=content,
            ),
            idempotency_key=f"result-{invocation_id}-{status}",
            actor="tool_worker",
            action_id=f"tool-worker-{invocation_id}",
        )

    def _commit_results(
        self,
        trajectory_id: str,
        expected_sequence: int,
        expected_head_hash: ContentHash | None,
        proposals: list[ProposedEvent],
    ) -> None:
        from rfsn_agent.store import StaleContextError

        try:
            self.store.commit_events(
                trajectory_id=TrajectoryId(trajectory_id),
                expected_sequence=expected_sequence,
                expected_head_hash=expected_head_hash,
                proposed_events=tuple(proposals),
            )
        except StaleContextError:
            # If the head moved, the pending invocations will be re-evaluated
            # on the next poll cycle.
            pass
