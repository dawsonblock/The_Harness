"""Agent runtime: context compiler + action loop over the event store."""

from __future__ import annotations

from typing import Protocol

from rfsn_agent.actions import (
    Action,
    ActionError,
    plan_events,
    validate_action,
)
from rfsn_agent.context import CompilerConfig, ContextPacket, TokenCounter, compile_context
from rfsn_agent.domain import HarnessSnapshot
from rfsn_agent.store import SQLiteEventStore


class Policy(Protocol):
    """A callable that selects an action given a context packet and snapshot."""

    def __call__(self, context: ContextPacket, snapshot: HarnessSnapshot) -> Action:
        ...


class Runtime:
    """Drives the action loop for a single trajectory.

    The runtime is deliberately thin: it compiles state into a context packet,
    asks the policy for an action, validates it, converts it to events, and
    appends those events to the store. All semantic state lives in the event
    log and derived snapshots.
    """

    def __init__(
        self,
        store: SQLiteEventStore,
        compiler_config: CompilerConfig,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self.store = store
        self.compiler_config = compiler_config
        self.token_counter = token_counter

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
        validate_action(action, snapshot, token_cost_estimate=token_cost_estimate)
        events = plan_events(action, snapshot, action_id=action_id, actor=actor)
        self.store.append_events(events)
        return self.store.get_latest_snapshot(trajectory_id)

    def step(
        self,
        trajectory_id: str,
        policy: Policy,
        *,
        action_id: str | None = None,
        token_cost_estimate: int = 0,
    ) -> HarnessSnapshot:
        """Compile context, ask the policy for an action, and execute it."""
        snapshot = self.store.get_latest_snapshot(trajectory_id)
        context = compile_context(
            snapshot, self.compiler_config, self.token_counter
        )
        action = policy(context, snapshot)
        if action_id is None:
            action_id = f"step-{snapshot.sequence}"
        return self.execute(
            trajectory_id,
            action,
            action_id=action_id,
            token_cost_estimate=token_cost_estimate,
        )

    def run(
        self,
        trajectory_id: str,
        policy: Policy,
        *,
        max_steps: int = 10,
        stop_on: tuple[type[Exception], ...] = (ActionError,),
    ) -> HarnessSnapshot:
        """Run the policy loop for up to ``max_steps`` iterations."""
        snapshot = self.store.get_latest_snapshot(trajectory_id)
        for step_idx in range(max_steps):
            try:
                snapshot = self.step(
                    trajectory_id,
                    policy,
                    action_id=f"run-{step_idx}",
                )
            except stop_on:
                break
        return snapshot
