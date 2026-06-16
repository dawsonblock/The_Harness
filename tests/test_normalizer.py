"""Tests for the ResultNormalizer."""

from __future__ import annotations

import tempfile
from pathlib import Path

from rfsn_agent.domain import BudgetLedger
from rfsn_agent.events import (
    ProposedEvent,
    ToolInvokedPayload,
    ToolResultReceivedPayload,
)
from rfsn_agent.normalizer import ResultNormalizer
from rfsn_agent.store import SQLiteEventStore


def test_normalizer_creates_candidates_for_successful_tool_results() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = SQLiteEventStore(Path(tmp) / "events.db")
        store.init_trajectory(
            "traj-1",
            budget=BudgetLedger(trajectory_id="traj-1", max_tokens=1000),
        )
        snap0 = store.get_latest_snapshot("traj-1")
        store.commit_events(
            trajectory_id="traj-1",
            expected_sequence=snap0.sequence,
            expected_head_hash=snap0.last_event_hash,
            proposed_events=(
                ProposedEvent(
                    event_type="tool_invoked",
                    payload=ToolInvokedPayload(
                        invocation_id="tool-1",
                        parent_task_id=None,
                        tool_name="read_file",
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
                        invocation_id="tool-1",
                        status="success",
                        content="result body",
                    ),
                    idempotency_key="idem-result-1",
                    actor="policy",
                    action_id="act-1",
                ),
            ),
        )

        normalizer = ResultNormalizer(store)
        normalizer.normalize("traj-1")

        snap = store.get_latest_snapshot("traj-1")
        assert len(snap.candidates) == 1
        assert snap.candidates[0].item_id == "auto-cand-tool-1"
        assert snap.candidates[0].content == "result body"
        assert snap.candidates[0].source_id == "tool-1"


def test_normalizer_ignores_failed_tool_results() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = SQLiteEventStore(Path(tmp) / "events.db")
        store.init_trajectory(
            "traj-1",
            budget=BudgetLedger(trajectory_id="traj-1", max_tokens=1000),
        )
        snap0 = store.get_latest_snapshot("traj-1")
        store.commit_events(
            trajectory_id="traj-1",
            expected_sequence=snap0.sequence,
            expected_head_hash=snap0.last_event_hash,
            proposed_events=(
                ProposedEvent(
                    event_type="tool_invoked",
                    payload=ToolInvokedPayload(
                        invocation_id="tool-1",
                        parent_task_id=None,
                        tool_name="read_file",
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
                        invocation_id="tool-1",
                        status="failure",
                        content="error",
                    ),
                    idempotency_key="idem-result-1",
                    actor="policy",
                    action_id="act-1",
                ),
            ),
        )

        normalizer = ResultNormalizer(store)
        normalizer.normalize("traj-1")

        snap = store.get_latest_snapshot("traj-1")
        assert len(snap.candidates) == 0


def test_normalizer_is_idempotent() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = SQLiteEventStore(Path(tmp) / "events.db")
        store.init_trajectory(
            "traj-1",
            budget=BudgetLedger(trajectory_id="traj-1", max_tokens=1000),
        )
        snap0 = store.get_latest_snapshot("traj-1")
        store.commit_events(
            trajectory_id="traj-1",
            expected_sequence=snap0.sequence,
            expected_head_hash=snap0.last_event_hash,
            proposed_events=(
                ProposedEvent(
                    event_type="tool_invoked",
                    payload=ToolInvokedPayload(
                        invocation_id="tool-1",
                        parent_task_id=None,
                        tool_name="read_file",
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
                        invocation_id="tool-1",
                        status="success",
                        content="result body",
                    ),
                    idempotency_key="idem-result-1",
                    actor="policy",
                    action_id="act-1",
                ),
            ),
        )

        normalizer = ResultNormalizer(store)
        normalizer.normalize("traj-1")
        normalizer.normalize("traj-1")

        snap = store.get_latest_snapshot("traj-1")
        assert len(snap.candidates) == 1
        assert snap.candidates[0].item_id == "auto-cand-tool-1"
