"""Tests for the asynchronous ToolWorker executor."""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

import pytest

from rfsn_agent.domain import BudgetLedger
from rfsn_agent.events import ProposedEvent, ToolInvokedPayload, ToolResultReceivedPayload
from rfsn_agent.security import SecurityProfile
from rfsn_agent.store import SQLiteEventStore
from rfsn_agent.tool_worker import ToolWorker


def _init_store(tmp: str) -> SQLiteEventStore:
    db_path = Path(tmp) / "events.db"
    store = SQLiteEventStore(db_path)
    store.init_trajectory("traj-1", budget=BudgetLedger(trajectory_id="traj-1", max_tokens=1000))
    return store


async def _commit_tool_invoked(
    store: SQLiteEventStore,
    invocation_id: str,
    tool_name: str,
    arguments: tuple[tuple[str, str], ...] = (),
    dependency_ids: tuple[str, ...] = (),
    deadline: str | None = None,
) -> None:
    from datetime import datetime

    snap = store.get_latest_snapshot("traj-1")
    dl = datetime.fromisoformat(deadline) if deadline else None
    store.commit_events(
        trajectory_id="traj-1",
        expected_sequence=snap.sequence,
        expected_head_hash=snap.last_event_hash,
        proposed_events=(
            ProposedEvent(
                event_type="tool_invoked",
                payload=ToolInvokedPayload(
                    invocation_id=invocation_id,
                    parent_task_id=None,
                    tool_name=tool_name,
                    arguments=arguments,
                    dependency_ids=dependency_ids,
                    deadline=dl,
                ),
                idempotency_key=f"idem-{invocation_id}",
                actor="policy",
                action_id="act-1",
            ),
        ),
    )


@pytest.mark.anyio
async def test_tool_worker_executes_read_file() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = _init_store(tmp)
        test_file = Path(tmp) / "test.txt"
        test_file.write_text("hello world")

        await _commit_tool_invoked(
            store, "tool-1", "read_file", (("source_id", str(test_file)),)
        )

        worker = ToolWorker(store)
        count = await worker.process_pending("traj-1")
        assert count == 1

        snap = store.get_latest_snapshot("traj-1")
        assert len(snap.tool_results) == 1
        assert snap.tool_results[0].content == "hello world"
        assert snap.tool_results[0].status.value == "success"


@pytest.mark.anyio
async def test_tool_worker_blocks_disallowed_tool() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = _init_store(tmp)
        await _commit_tool_invoked(store, "tool-1", "web_search", (("query", "foo"),))

        profile = SecurityProfile(allowed_tool_names=frozenset())
        worker = ToolWorker(store, security_profile=profile)
        count = await worker.process_pending("traj-1")
        assert count == 1

        snap = store.get_latest_snapshot("traj-1")
        assert snap.tool_results[0].status.value == "failure"
        assert "not allowed" in snap.tool_results[0].content


@pytest.mark.anyio
async def test_tool_worker_respects_forbidden_path() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = _init_store(tmp)
        await _commit_tool_invoked(
            store, "tool-1", "read_file", (("source_id", "../secret.txt"),)
        )

        import re

        profile = SecurityProfile(
            allowed_tool_names=frozenset({"read_file"}),
            forbidden_path_pattern=re.compile(r"\.\."),
        )
        worker = ToolWorker(store, security_profile=profile)
        count = await worker.process_pending("traj-1")
        assert count == 1

        snap = store.get_latest_snapshot("traj-1")
        assert snap.tool_results[0].status.value == "failure"
        assert "outside" in snap.tool_results[0].content


@pytest.mark.anyio
async def test_tool_worker_enforces_workspace_jail() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = _init_store(tmp)
        root = Path(tmp)
        outside = Path(tmp).parent / "secret.txt"
        await _commit_tool_invoked(
            store, "tool-1", "read_file", (("source_id", str(outside)),)
        )

        profile = SecurityProfile(
            allowed_tool_names=frozenset({"read_file"}),
            allowed_workspace_root=root,
        )
        worker = ToolWorker(store, security_profile=profile)
        count = await worker.process_pending("traj-1")
        assert count == 1

        snap = store.get_latest_snapshot("traj-1")
        assert snap.tool_results[0].status.value == "failure"
        assert "outside" in snap.tool_results[0].content


@pytest.mark.anyio
async def test_tool_worker_blocks_symlink_escape() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = _init_store(tmp)
        root = Path(tmp) / "workspace"
        root.mkdir()
        outside = Path(tmp) / "secret.txt"
        outside.write_text("secret")
        link = root / "link.txt"
        os.symlink(outside, link)
        await _commit_tool_invoked(
            store, "tool-1", "read_file", (("source_id", str(link)),)
        )

        profile = SecurityProfile(
            allowed_tool_names=frozenset({"read_file"}),
            allowed_workspace_root=root,
        )
        worker = ToolWorker(store, security_profile=profile)
        count = await worker.process_pending("traj-1")
        assert count == 1

        snap = store.get_latest_snapshot("traj-1")
        assert snap.tool_results[0].status.value == "failure"
        assert "outside" in snap.tool_results[0].content


@pytest.mark.anyio
async def test_tool_worker_enforces_dependencies() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = _init_store(tmp)
        await _commit_tool_invoked(store, "tool-1", "web_search", dependency_ids=())
        await _commit_tool_invoked(
            store, "tool-2", "read_file", dependency_ids=("tool-1",)
        )

        worker = ToolWorker(store)
        # Both are pending; tool-2 depends on tool-1.
        count = await worker.process_pending("traj-1")
        assert count == 1  # Only tool-1 succeeds.

        snap = store.get_latest_snapshot("traj-1")
        assert len(snap.tool_results) == 1
        assert snap.tool_results[0].invocation_id == "tool-1"

        # Now tool-1 has a result; process again.
        count = await worker.process_pending("traj-1")
        assert count == 1

        snap = store.get_latest_snapshot("traj-1")
        assert len(snap.tool_results) == 2
        assert snap.tool_results[1].invocation_id == "tool-2"


@pytest.mark.anyio
async def test_tool_worker_dependency_failure_blocks_dependent() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = _init_store(tmp)
        # Commit tool-1 invocation + a failed result manually.
        snap = store.get_latest_snapshot("traj-1")
        store.commit_events(
            trajectory_id="traj-1",
            expected_sequence=snap.sequence,
            expected_head_hash=snap.last_event_hash,
            proposed_events=(
                ProposedEvent(
                    event_type="tool_invoked",
                    payload=ToolInvokedPayload(
                        invocation_id="tool-1",
                        parent_task_id=None,
                        tool_name="web_search",
                        arguments=(),
                        dependency_ids=(),
                        deadline=None,
                    ),
                    idempotency_key="idem-tool-1",
                    actor="policy",
                    action_id="act-1",
                ),
                ProposedEvent(
                    event_type="tool_result_received",
                    payload=ToolResultReceivedPayload(
                        invocation_id="tool-1", status="failure", content="network error"
                    ),
                    idempotency_key="idem-result-1",
                    actor="policy",
                    action_id="act-1",
                ),
            ),
        )
        await _commit_tool_invoked(
            store, "tool-2", "read_file", dependency_ids=("tool-1",)
        )

        worker = ToolWorker(store)
        count = await worker.process_pending("traj-1")
        # tool-2 is skipped because its dependency (tool-1) failed, not recorded.
        assert count == 0

        snap = store.get_latest_snapshot("traj-1")
        # Only tool-1's original failure result exists.
        assert len(snap.tool_results) == 1
        assert snap.tool_results[0].invocation_id == "tool-1"


@pytest.mark.anyio
async def test_tool_worker_deadline_timeout() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = _init_store(tmp)
        from datetime import UTC, datetime, timedelta

        past = datetime.now(UTC) - timedelta(seconds=10)
        await _commit_tool_invoked(
            store,
            "tool-1",
            "web_search",
            deadline=past.isoformat(),
        )

        worker = ToolWorker(store)
        count = await worker.process_pending("traj-1")
        assert count == 1

        snap = store.get_latest_snapshot("traj-1")
        assert snap.tool_results[0].status.value == "timeout"
        assert "deadline" in snap.tool_results[0].content


@pytest.mark.anyio
async def test_tool_worker_lease_prevents_duplicate_execution() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = _init_store(tmp)
        test_file = Path(tmp) / "slow.txt"
        test_file.write_text("slow data")

        await _commit_tool_invoked(
            store, "tool-1", "read_file", (("source_id", str(test_file)),)
        )

        worker = ToolWorker(store)
        # Simulate two concurrent process_pending calls.
        results = await asyncio.gather(
            worker.process_pending("traj-1"),
            worker.process_pending("traj-1"),
        )
        # Only one should actually commit the result.
        assert sum(results) == 1

        snap = store.get_latest_snapshot("traj-1")
        assert len(snap.tool_results) == 1


@pytest.mark.anyio
async def test_tool_worker_unknown_tool_returns_failure() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = _init_store(tmp)
        await _commit_tool_invoked(store, "tool-1", "unknown_tool")

        worker = ToolWorker(store)
        count = await worker.process_pending("traj-1")
        assert count == 1

        snap = store.get_latest_snapshot("traj-1")
        assert snap.tool_results[0].status.value == "failure"
        assert "Unknown tool" in snap.tool_results[0].content


@pytest.mark.anyio
async def test_tool_worker_custom_registry() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = _init_store(tmp)

        async def my_tool(name: str, args: tuple[tuple[str, str], ...]) -> str:
            return "custom-result"

        await _commit_tool_invoked(store, "tool-1", "my_tool")

        worker = ToolWorker(store, tool_registry={"my_tool": my_tool})
        count = await worker.process_pending("traj-1")
        assert count == 1

        snap = store.get_latest_snapshot("traj-1")
        assert snap.tool_results[0].content == "custom-result"


@pytest.mark.anyio
async def test_tool_worker_start_stop_poll_loop() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = _init_store(tmp)
        worker = ToolWorker(store)
        await worker.start(poll_interval_seconds=0.1)
        await asyncio.sleep(0.05)  # Let it start.
        await worker.stop()
        assert not worker._running
