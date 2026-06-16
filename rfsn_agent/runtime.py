"""Agent runtime: context compiler + action loop over the event store."""

from __future__ import annotations

import uuid
from typing import Protocol

from rfsn_agent.actions import (
    Action,
    ActionError,
    plan_events,
    validate_action,
)
from rfsn_agent.context import CompilerConfig, ContextPacket, TokenCounter, compile_context
from rfsn_agent.domain import HarnessSnapshot
from rfsn_agent.security import SecurityProfile
from rfsn_agent.store import SQLiteEventStore, StaleContextError
from rfsn_agent.types import TrajectoryId


class Policy(Protocol):
    """A callable that selects an action given a context packet and snapshot."""

    def __call__(self, context: ContextPacket, snapshot: HarnessSnapshot) -> Action:
        ...


class Runtime:
    """Drives the action loop for a single trajectory.

    The runtime is deliberately thin: it compiles state into a context packet,
    asks the policy for an action, validates it, converts it to proposed
    events, and commits those events to the store. All semantic state lives in
    the event log and derived snapshots.
    """

    def __init__(
        self,
        store: SQLiteEventStore,
        compiler_config: CompilerConfig,
        token_counter: TokenCounter | None = None,
        security_profile: SecurityProfile | None = None,
    ) -> None:
        self.store = store
        self.compiler_config = compiler_config
        self.token_counter = token_counter
        self.security_profile = security_profile

    def execute(
        self,
        trajectory_id: str,
        action: Action,
        *,
        action_id: str,
        actor: str = "policy",
        token_cost_estimate: int = 0,
    ) -> HarnessSnapshot:
        """Validate and execute one action, returning the new snapshot."""
        snapshot = self.store.get_latest_snapshot(trajectory_id)
        validate_action(
            action,
            snapshot,
            token_cost_estimate=token_cost_estimate,
            allowed_tool_names=(
                self.security_profile.allowed_tool_names
                if self.security_profile is not None
                else None
            ),
            forbidden_path_pattern=(
                self.security_profile.forbidden_path_pattern
                if self.security_profile is not None
                else None
            ),
        )
        proposed = plan_events(action, snapshot, action_id=action_id, actor=actor)
        self.store.commit_events(
            trajectory_id=TrajectoryId(trajectory_id),
            expected_sequence=snapshot.sequence,
            expected_head_hash=snapshot.last_event_hash,
            proposed_events=tuple(proposed),
        )
        return self.store.get_latest_snapshot(trajectory_id)

    def step(
        self,
        trajectory_id: str,
        policy: Policy,
        *,
        action_id: str | None = None,
        token_cost_estimate: int = 0,
        max_retries: int = 3,
    ) -> HarnessSnapshot:
        """Compile context, ask the policy for an action, and execute it.

        If the trajectory head changes between context compilation and commit,
        the step is retried with fresh context up to ``max_retries`` times.
        """
        for _ in range(max_retries):
            snapshot = self.store.get_latest_snapshot(trajectory_id)
            context = compile_context(
                snapshot, self.compiler_config, self.token_counter
            )
            action = policy(context, snapshot)
            if action_id is None:
                action_id = f"step-{uuid.uuid4().hex}"
            try:
                validate_action(
                    action,
                    snapshot,
                    token_cost_estimate=token_cost_estimate,
                    allowed_tool_names=(
                        self.security_profile.allowed_tool_names
                        if self.security_profile is not None
                        else None
                    ),
                    forbidden_path_pattern=(
                        self.security_profile.forbidden_path_pattern
                        if self.security_profile is not None
                        else None
                    ),
                )
                proposed = plan_events(action, snapshot, action_id=action_id, actor="policy")
                self.store.commit_events(
                    trajectory_id=TrajectoryId(trajectory_id),
                    expected_sequence=snapshot.sequence,
                    expected_head_hash=snapshot.last_event_hash,
                    proposed_events=tuple(proposed),
                )
                return self.store.get_latest_snapshot(trajectory_id)
            except StaleContextError:
                continue
        raise StaleContextError(
            f"Could not commit step for {trajectory_id}: head changed too many times"
        )

    def run(
        self,
        trajectory_id: str,
        policy: Policy,
        *,
        max_steps: int = 10,
        stop_on: tuple[type[Exception], ...] = (ActionError,),
        stop_on_submit: bool = True,
    ) -> HarnessSnapshot:
        """Run the policy loop for up to ``max_steps`` iterations.

        If ``stop_on_submit`` is True (the default), the loop terminates
        cleanly as soon as the policy emits a :class:`SubmitAction`,
        signalling that the agent considers the task complete.
        """
        run_id = uuid.uuid4().hex
        snapshot = self.store.get_latest_snapshot(trajectory_id)
        for step_idx in range(max_steps):
            try:
                snapshot = self.step(
                    trajectory_id,
                    policy,
                    action_id=f"run-{run_id}-step-{step_idx}",
                )
            except stop_on:
                break
            # Terminal-state: a SubmitAction means the agent is done.
            if stop_on_submit and snapshot.submissions:
                latest_submission = snapshot.submissions[-1]
                # Only break if this submission was produced by the current run.
                if latest_submission.provenance.action_id.startswith(f"run-{run_id}"):
                    break
        return snapshot
