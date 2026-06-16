"""Tests for the pure event reducer."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from rfsn_agent.domain import (
    BudgetLedger,
    CandidateItem,
    Claim,
    CuratedItem,
    EvidenceLink,
    HarnessSnapshot,
    TaskNode,
)
from rfsn_agent.events import (
    ActionCommittedPayload,
    ClaimRevisedPayload,
    ContextPrunedPayload,
    EvidenceCuratedPayload,
    EvidenceVerifiedPayload,
    HarnessEvent,
    SnapshotCheckpointedPayload,
    SubmissionRecordedPayload,
    TaskCompletedPayload,
    TaskDecomposedPayload,
    ToolInvokedPayload,
    ToolResultReceivedPayload,
)
from rfsn_agent.reducer import InvariantError, ReducerError, reduce_event
from rfsn_agent.types import ClaimStatus, TaskStatus, VerificationResult


def _empty_snapshot(trajectory_id: str = "traj-1") -> HarnessSnapshot:
    return HarnessSnapshot.create(
        trajectory_id=trajectory_id,
        epoch_id="epoch-0",
        sequence=0,
        budget=BudgetLedger(trajectory_id=trajectory_id, max_tokens=1000),
    )


def _event(
    snapshot: HarnessSnapshot,
    sequence: int,
    event_type: str,
    payload: object,
    idempotency_key: str,
    event_id: str | None = None,
) -> HarnessEvent:
    return HarnessEvent.create(
        event_id=event_id or f"evt-{sequence}",
        trajectory_id=snapshot.trajectory_id,
        sequence=sequence,
        event_type=event_type,
        payload=payload,  # type: ignore[arg-type]
        idempotency_key=idempotency_key,
        actor="policy",
        action_id="act-1",
    )


def test_action_committed_advances_sequence_only() -> None:
    snap = _empty_snapshot()
    event = _event(
        snap,
        1,
        "action_committed",
        ActionCommittedPayload(action_type="search", action_params=()),
        "idem-1",
    )
    next_snap = reduce_event(snap, event)
    assert next_snap.sequence == 1
    assert next_snap.epoch_id == snap.epoch_id


def test_tool_invoked_then_result_received() -> None:
    snap = _empty_snapshot()
    invoked = _event(
        snap,
        1,
        "tool_invoked",
        ToolInvokedPayload(
            invocation_id="tool-1",
            parent_task_id=None,
            tool_name="read_file",
            arguments=(("path", "/tmp/foo"),),
            dependency_ids=(),
            deadline=None,
        ),
        "idem-1",
    )
    snap2 = reduce_event(snap, invoked)
    assert len(snap2.tool_invocations) == 1
    assert snap2.tool_invocations[0].tool_name == "read_file"

    result = _event(
        snap2,
        2,
        "tool_result_received",
        ToolResultReceivedPayload(
            invocation_id="tool-1", status="success", content="hello"
        ),
        "idem-2",
    )
    snap3 = reduce_event(snap2, result)
    assert len(snap3.tool_results) == 1
    assert snap3.tool_results[0].content == "hello"


def test_tool_result_for_unknown_invocation_fails() -> None:
    snap = _empty_snapshot()
    result = _event(
        snap,
        1,
        "tool_result_received",
        ToolResultReceivedPayload(
            invocation_id="tool-1", status="success", content="hello"
        ),
        "idem-1",
    )
    with pytest.raises(InvariantError, match="unknown invocation"):
        reduce_event(snap, result)


def test_duplicate_tool_result_fails() -> None:
    snap = _empty_snapshot()
    snap2 = reduce_event(
        snap,
        _event(
            snap,
            1,
            "tool_invoked",
            ToolInvokedPayload(
                invocation_id="tool-1",
                parent_task_id=None,
                tool_name="read",
                arguments=(),
                dependency_ids=(),
                deadline=None,
            ),
            "idem-1",
        ),
    )
    snap3 = reduce_event(
        snap2,
        _event(
            snap2,
            2,
            "tool_result_received",
            ToolResultReceivedPayload(
                invocation_id="tool-1", status="success", content="ok"
            ),
            "idem-2",
        ),
    )
    with pytest.raises(InvariantError, match="Duplicate ToolResult"):
        reduce_event(
            snap3,
            _event(
                snap3,
                3,
                "tool_result_received",
                ToolResultReceivedPayload(
                    invocation_id="tool-1", status="success", content="ok2"
                ),
                "idem-3",
            ),
        )


def test_evidence_curated_from_candidates() -> None:
    snap = HarnessSnapshot.create(
        trajectory_id="traj-1",
        epoch_id="epoch-0",
        sequence=0,
        candidates=(
            CandidateItem.create(
                item_id="cand-1",
                trajectory_id="traj-1",
                source_id="src-1",
                retrieval_query="q",
                content="candidate body",
            ),
        ),
    )
    event = _event(
        snap,
        1,
        "evidence_curated",
        EvidenceCuratedPayload(
            candidate_ids=("cand-1",),
            curated_item_id="cur-1",
            content="curated body",
            priority=3,
            source_ids=("src-1",),
        ),
        "idem-1",
    )
    snap2 = reduce_event(snap, event)
    assert len(snap2.curated_items) == 1
    assert snap2.curated_items[0].content == "curated body"
    assert snap2.curated_items[0].priority == 3


def test_evidence_curate_unknown_candidate_fails() -> None:
    snap = _empty_snapshot()
    event = _event(
        snap,
        1,
        "evidence_curated",
        EvidenceCuratedPayload(
            candidate_ids=("cand-1",),
            curated_item_id="cur-1",
            content="curated body",
            priority=1,
            source_ids=(),
        ),
        "idem-1",
    )
    with pytest.raises(InvariantError, match="unknown candidates"):
        reduce_event(snap, event)


def test_claim_revised() -> None:
    snap = HarnessSnapshot.create(
        trajectory_id="traj-1",
        epoch_id="epoch-0",
        sequence=0,
        claims=(
            Claim.create(
                claim_id="claim-1",
                trajectory_id="traj-1",
                content="original",
            ),
        ),
    )
    event = _event(
        snap,
        1,
        "claim_revised",
        ClaimRevisedPayload(
            claim_id="claim-1",
            new_content="revised",
            new_status=ClaimStatus.VERIFIED,
        ),
        "idem-1",
    )
    snap2 = reduce_event(snap, event)
    assert snap2.claims[0].content == "revised"
    assert snap2.claims[0].status == ClaimStatus.VERIFIED


def test_evidence_verified() -> None:
    snap = HarnessSnapshot.create(
        trajectory_id="traj-1",
        epoch_id="epoch-0",
        sequence=0,
        claims=(
            Claim.create(
                claim_id="claim-1", trajectory_id="traj-1", content="c"
            ),
        ),
        curated_items=(
            CuratedItem.create(
                item_id="cur-1",
                trajectory_id="traj-1",
                candidate_ids=(),
                content="evidence",
            ),
        ),
        evidence_links=(
            EvidenceLink(
                link_id="link-1",
                trajectory_id="traj-1",
                claim_id="claim-1",
                curated_item_id="cur-1",
                relationship="supports",
                strength=0.9,
            ),
        ),
    )
    event = _event(
        snap,
        1,
        "evidence_verified",
        EvidenceVerifiedPayload(
            link_id="link-1",
            verification_id="ver-1",
            result=VerificationResult.CONFIRMED,
            details="two sources",
        ),
        "idem-1",
    )
    snap2 = reduce_event(snap, event)
    assert snap2.evidence_links[0].verified is True
    assert snap2.evidence_links[0].verification_id == "ver-1"
    assert len(snap2.verification_records) == 1


def test_task_decomposed_and_completed() -> None:
    snap = _empty_snapshot()
    decomp = _event(
        snap,
        1,
        "task_decomposed",
        TaskDecomposedPayload(
            parent_task_id=None,
            task_id="task-1",
            description="do it",
            dependency_ids=(),
        ),
        "idem-1",
    )
    snap2 = reduce_event(snap, decomp)
    assert len(snap2.tasks) == 1
    assert snap2.tasks[0].status == TaskStatus.PENDING

    complete = _event(
        snap2,
        2,
        "task_completed",
        TaskCompletedPayload(task_id="task-1"),
        "idem-2",
    )
    snap3 = reduce_event(snap2, complete)
    assert snap3.tasks[0].status == TaskStatus.COMPLETED
    assert snap3.tasks[0].completed_at is not None


def test_complete_already_completed_task_fails() -> None:
    snap = HarnessSnapshot.create(
        trajectory_id="traj-1",
        epoch_id="epoch-0",
        sequence=0,
        tasks=(
            TaskNode(
                task_id="task-1",
                trajectory_id="traj-1",
                parent_id=None,
                description="done",
                status=TaskStatus.COMPLETED,
                completed_at=datetime.now(UTC),
            ),
        ),
    )
    event = _event(
        snap,
        1,
        "task_completed",
        TaskCompletedPayload(task_id="task-1"),
        "idem-1",
    )
    with pytest.raises(InvariantError, match="already completed"):
        reduce_event(snap, event)


def test_context_pruned_creates_new_epoch() -> None:
    snap = HarnessSnapshot.create(
        trajectory_id="traj-1",
        epoch_id="epoch-0",
        sequence=0,
        curated_items=(
            CuratedItem.create(
                item_id="cur-1",
                trajectory_id="traj-1",
                candidate_ids=(),
                content="keep",
            ),
            CuratedItem.create(
                item_id="cur-2",
                trajectory_id="traj-1",
                candidate_ids=(),
                content="discard",
            ),
        ),
    )
    event = _event(
        snap,
        1,
        "context_pruned",
        ContextPrunedPayload(
            retained_item_ids=("cur-1",), new_epoch_id="epoch-1"
        ),
        "idem-1",
    )
    snap2 = reduce_event(snap, event)
    assert snap2.epoch_id == "epoch-1"
    assert [c.item_id for c in snap2.curated_items] == ["cur-1"]


def test_submission_recorded() -> None:
    snap = _empty_snapshot()
    event = _event(
        snap,
        1,
        "submission_recorded",
        SubmissionRecordedPayload(
            submission_id="sub-1", content="answer", source_ids=("src-1",)
        ),
        "idem-1",
    )
    snap2 = reduce_event(snap, event)
    assert len(snap2.submissions) == 1
    assert snap2.submissions[0].content == "answer"


def test_snapshot_checkpointed_validates_hash() -> None:
    snap = _empty_snapshot()
    event = _event(
        snap,
        1,
        "snapshot_checkpointed",
        SnapshotCheckpointedPayload(
            snapshot_sequence=0, snapshot_hash=snap.compute_state_hash()
        ),
        "idem-1",
    )
    snap2 = reduce_event(snap, event)
    assert snap2.sequence == 1

    bad = _event(
        snap2,
        2,
        "snapshot_checkpointed",
        SnapshotCheckpointedPayload(snapshot_sequence=1, snapshot_hash="bad"),
        "idem-2",
    )
    with pytest.raises(InvariantError, match="Checkpoint hash mismatch"):
        reduce_event(snap2, bad)


def test_sequence_gap_fails() -> None:
    snap = _empty_snapshot()
    event = _event(
        snap,
        5,
        "action_committed",
        ActionCommittedPayload(action_type="search", action_params=()),
        "idem-1",
    )
    with pytest.raises(InvariantError, match="Sequence gap"):
        reduce_event(snap, event)


def test_trajectory_mismatch_fails() -> None:
    snap = _empty_snapshot("traj-1")
    event = HarnessEvent.create(
        event_id="evt-1",
        trajectory_id="traj-2",
        sequence=1,
        event_type="action_committed",
        payload=ActionCommittedPayload(action_type="search", action_params=()),
        idempotency_key="idem-1",
    )
    with pytest.raises(InvariantError, match="Trajectory mismatch"):
        reduce_event(snap, event)


def test_idempotent_replay_returns_same_snapshot() -> None:
    snap = _empty_snapshot()
    event = _event(
        snap,
        1,
        "action_committed",
        ActionCommittedPayload(action_type="search", action_params=()),
        "idem-1",
    )
    snap2 = reduce_event(snap, event)
    snap3 = reduce_event(snap2, event)
    assert snap3.sequence == snap2.sequence
    assert snap3.state_hash == snap2.state_hash


def test_unknown_event_type_fails() -> None:
    snap = _empty_snapshot()
    event = _event(
        snap,
        1,
        "unknown_event",
        ActionCommittedPayload(action_type="search", action_params=()),
        "idem-1",
    )
    with pytest.raises(ReducerError, match="Unknown event type"):
        reduce_event(snap, event)


def test_reducer_is_deterministic() -> None:
    snap = _empty_snapshot()
    event = _event(
        snap,
        1,
        "action_committed",
        ActionCommittedPayload(action_type="search", action_params=()),
        "idem-1",
    )
    a = reduce_event(snap, event)
    b = reduce_event(snap, event)
    assert a.state_hash == b.state_hash
