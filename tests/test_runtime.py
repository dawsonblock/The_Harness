"""Tests for the agent runtime action loop."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from rfsn_agent.actions import (
    Action,
    ActionError,
    DecomposeAction,
    SearchAction,
    SubmitAction,
)
from rfsn_agent.context import CompilerConfig
from rfsn_agent.domain import BudgetLedger, HarnessSnapshot
from rfsn_agent.runtime import Runtime
from rfsn_agent.store import SQLiteEventStore


def _make_runtime(
    tmp: str, max_tokens: int = 1000, max_tool_calls: int | None = None
) -> tuple[Runtime, str]:
    db_path = Path(tmp) / "events.db"
    store = SQLiteEventStore(db_path)
    trajectory_id = "traj-1"
    store.init_trajectory(
        trajectory_id,
        budget=BudgetLedger(
            trajectory_id=trajectory_id,
            max_tokens=max_tokens,
            max_tool_calls=max_tool_calls,
        ),
    )
    config = CompilerConfig(max_total_tokens=500)
    runtime = Runtime(store, config)
    return runtime, trajectory_id


def test_runtime_execute_decompose() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        runtime, traj = _make_runtime(tmp)
        action = DecomposeAction(
            task_id="task-1",
            parent_task_id=None,
            description="find the bug",
            dependency_ids=(),
        )
        snap = runtime.execute(traj, action, action_id="act-1")
        assert snap.sequence == 1
        assert len(snap.tasks) == 1


def test_runtime_step_with_policy() -> None:
    def policy(context: object, snapshot: HarnessSnapshot) -> Action:
        return DecomposeAction(
            task_id="task-1",
            parent_task_id=None,
            description="do it",
            dependency_ids=(),
        )

    with tempfile.TemporaryDirectory() as tmp:
        runtime, traj = _make_runtime(tmp)
        snap = runtime.step(traj, policy)
        assert snap.sequence == 1


def test_runtime_run_multiple_steps() -> None:
    steps = {"count": 0}

    def policy(context: object, snapshot: HarnessSnapshot) -> Action:
        steps["count"] += 1
        return DecomposeAction(
            task_id=f"task-{steps['count']}",
            parent_task_id=None,
            description=f"step {steps['count']}",
            dependency_ids=(),
        )

    with tempfile.TemporaryDirectory() as tmp:
        runtime, traj = _make_runtime(tmp)
        snap = runtime.run(traj, policy, max_steps=3)
        assert snap.sequence == 3
        assert len(snap.tasks) == 3


def test_runtime_action_validation_blocks_bad_action() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        runtime, traj = _make_runtime(tmp)
        # Search with empty query violates safety.
        action = SearchAction(query="  ")
        with pytest.raises(ActionError):
            runtime.execute(traj, action, action_id="act-1")
        snap = runtime.store.get_latest_snapshot(traj)
        assert snap.sequence == 0


def test_runtime_submit_action() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        runtime, traj = _make_runtime(tmp)
        action = SubmitAction(
            submission_id="sub-1", content="final answer", source_ids=("src-1",)
        )
        snap = runtime.execute(traj, action, action_id="act-1")
        assert snap.sequence == 1
        assert len(snap.submissions) == 1


def test_runtime_respects_tool_call_budget() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        runtime, traj = _make_runtime(tmp, max_tool_calls=1)
        # First search consumes tool budget.
        runtime.execute(traj, SearchAction(query="first"), action_id="act-1")
        # Second search exceeds budget and should stop the run.
        def policy(context: object, snapshot: HarnessSnapshot) -> Action:
            return SearchAction(query="second")

        snap = runtime.run(traj, policy, max_steps=2)
        # The run stops on BudgetError after the first failed step.
        assert snap.sequence == 1


def test_runtime_rerun_uses_unique_run_ids() -> None:
    """Two separate Runtime.run() calls should not collide on idempotency keys."""
    calls = {"count": 0}

    def policy(context: object, snapshot: HarnessSnapshot) -> Action:
        calls["count"] += 1
        return DecomposeAction(
            task_id=f"task-{calls['count']}",
            parent_task_id=None,
            description=f"step {calls['count']}",
            dependency_ids=(),
        )

    with tempfile.TemporaryDirectory() as tmp:
        runtime, traj = _make_runtime(tmp)
        snap1 = runtime.run(traj, policy, max_steps=2)
        assert snap1.sequence == 2
        snap2 = runtime.run(traj, policy, max_steps=2)
        assert snap2.sequence == 4
        assert len(snap2.tasks) == 4
