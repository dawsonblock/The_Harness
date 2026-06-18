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
    CandidateAddedPayload,
    ClaimCreatedPayload,
    ClaimRevisedPayload,
    ContextPrunedPayload,
    EvidenceCuratedPayload,
    EvidenceLinkedPayload,
    EvidenceVerifiedPayload,
    HarnessEvent,
    SnapshotCheckpointedPayload,
    SubmissionRecordedPayload,
    TaskCompletedPayload,
    TaskDecomposedPayload,
    ToolInvokedPayload,
    ToolResultReceivedPayload,
)
from rfsn_agent.reducer import InvariantError, reduce_event
from rfsn_agent.types import (
    ClaimStatus,
    ContentHash,
    TaskStatus,
    VerificationResult,
    VerificationStatus,
)


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
    previous_hash = snapshot.last_event_hash
    return HarnessEvent.create(
        event_id=event_id or f"evt-{sequence}",
        trajectory_id=snapshot.trajectory_id,
        sequence=sequence,
        event_type=event_type,
        payload=payload,  # type: ignore[arg-type]
        idempotency_key=idempotency_key,
        previous_event_hash=previous_hash,
        actor="policy",
        action_id="act-1",
    )


def test_reducer_uses_event_timestamps_for_created_domain_objects() -> None:
    snap = HarnessSnapshot.create(
        trajectory_id="traj-1",
        epoch_id="epoch-0",
        sequence=0,
        budget=BudgetLedger(trajectory_id="traj-1", max_tokens=1000),
    )
    fixed_time = datetime(2030, 1, 1, tzinfo=UTC)
    event = HarnessEvent.create(
        event_id="evt-fixed",
        trajectory_id="traj-1",
        sequence=1,
        event_type="task_decomposed",
        payload=TaskDecomposedPayload(
            parent_task_id=None,
            task_id="task-1",
            description="task",
            dependency_ids=(),
        ),
        idempotency_key="idem-1",
        previous_event_hash=None,
        created_at=fixed_time,
        actor="policy",
        action_id="act-1",
    )

    next_snap = reduce_event(snap, event)

    assert next_snap.tasks[0].created_at == fixed_time


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
    assert next_snap.last_event_hash == event.header.event_hash


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
    assert snap2.evidence_links[0].current_status == VerificationStatus.VERIFIED
    assert snap2.evidence_links[0].verification_id == "ver-1"
    assert len(snap2.verification_records) == 1
    assert snap2.verification_records[0].link_id == "link-1"


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
        previous_event_hash=None,
    )
    with pytest.raises(InvariantError, match="Trajectory mismatch"):
        reduce_event(snap, event)


def test_replaying_same_event_fails_sequence_check() -> None:
    snap = _empty_snapshot()
    event = _event(
        snap,
        1,
        "action_committed",
        ActionCommittedPayload(action_type="search", action_params=()),
        "idem-1",
    )
    snap2 = reduce_event(snap, event)
    with pytest.raises(InvariantError, match="Sequence gap"):
        reduce_event(snap2, event)


def test_event_type_payload_mismatch_fails() -> None:
    snap = _empty_snapshot()
    with pytest.raises(ValueError, match="does not match payload type"):
        HarnessEvent.create(
            event_id="evt-1",
            trajectory_id=snap.trajectory_id,
            sequence=1,
            event_type="tool_invoked",
            payload=ActionCommittedPayload(action_type="search", action_params=()),
            idempotency_key="idem-1",
            previous_event_hash=snap.last_event_hash,
        )


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


def test_genesis_event_requires_no_previous_hash() -> None:
    snap = _empty_snapshot()
    event = HarnessEvent.create(
        event_id="evt-1",
        trajectory_id="traj-1",
        sequence=1,
        event_type="action_committed",
        payload=ActionCommittedPayload(action_type="search", action_params=()),
        idempotency_key="idem-1",
        previous_event_hash="a" * 64,
    )
    with pytest.raises(InvariantError, match="First event must not have a previous hash"):
        reduce_event(snap, event)


def test_non_genesis_event_requires_previous_hash() -> None:
    snap = _empty_snapshot()
    first = _event(
        snap,
        1,
        "action_committed",
        ActionCommittedPayload(action_type="search", action_params=()),
        "idem-1",
    )
    snap2 = reduce_event(snap, first)
    second = HarnessEvent.create(
        event_id="evt-2",
        trajectory_id="traj-1",
        sequence=2,
        event_type="action_committed",
        payload=ActionCommittedPayload(action_type="search", action_params=()),
        idempotency_key="idem-2",
        previous_event_hash=None,
    )
    with pytest.raises(InvariantError, match="Non-genesis event requires a previous hash"):
        reduce_event(snap2, second)


def test_wrong_previous_hash_fails() -> None:
    snap = _empty_snapshot()
    first = _event(
        snap,
        1,
        "action_committed",
        ActionCommittedPayload(action_type="search", action_params=()),
        "idem-1",
    )
    snap2 = reduce_event(snap, first)
    second = HarnessEvent.create(
        event_id="evt-2",
        trajectory_id="traj-1",
        sequence=2,
        event_type="action_committed",
        payload=ActionCommittedPayload(action_type="search", action_params=()),
        idempotency_key="idem-2",
        previous_event_hash="b" * 64,
    )
    with pytest.raises(InvariantError, match="Event chain mismatch"):
        reduce_event(snap2, second)


def test_candidate_added() -> None:
    snap = _empty_snapshot()
    event = _event(
        snap,
        1,
        "candidate_added",
        CandidateAddedPayload(
            item_id="cand-1",
            trajectory_id="traj-1",
            source_id="src-1",
            retrieval_query="q",
            content="candidate body",
        ),
        "idem-1",
    )
    snap2 = reduce_event(snap, event)
    assert len(snap2.candidates) == 1
    assert snap2.candidates[0].item_id == "cand-1"
    assert snap2.candidates[0].content == "candidate body"


def test_duplicate_candidate_added_fails() -> None:
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
        "candidate_added",
        CandidateAddedPayload(
            item_id="cand-1",
            trajectory_id="traj-1",
            source_id="src-1",
            retrieval_query="q",
            content="candidate body",
        ),
        "idem-1",
    )
    with pytest.raises(InvariantError, match="Duplicate CandidateItem id"):
        reduce_event(snap, event)


def test_claim_created() -> None:
    snap = _empty_snapshot()
    event = _event(
        snap,
        1,
        "claim_created",
        ClaimCreatedPayload(
            claim_id="claim-1",
            trajectory_id="traj-1",
            content="claim body",
        ),
        "idem-1",
    )
    snap2 = reduce_event(snap, event)
    assert len(snap2.claims) == 1
    assert snap2.claims[0].claim_id == "claim-1"
    assert snap2.claims[0].content == "claim body"
    assert snap2.claims[0].status == ClaimStatus.STATED


def test_duplicate_claim_created_fails() -> None:
    snap = HarnessSnapshot.create(
        trajectory_id="traj-1",
        epoch_id="epoch-0",
        sequence=0,
        claims=(
            Claim.create(
                claim_id="claim-1",
                trajectory_id="traj-1",
                content="claim body",
            ),
        ),
    )
    event = _event(
        snap,
        1,
        "claim_created",
        ClaimCreatedPayload(
            claim_id="claim-1",
            trajectory_id="traj-1",
            content="claim body",
        ),
        "idem-1",
    )
    with pytest.raises(InvariantError, match="Duplicate Claim id"):
        reduce_event(snap, event)


def test_evidence_linked() -> None:
    snap = HarnessSnapshot.create(
        trajectory_id="traj-1",
        epoch_id="epoch-0",
        sequence=0,
        claims=(
            Claim.create(
                claim_id="claim-1",
                trajectory_id="traj-1",
                content="foo",
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
    )
    event = _event(
        snap,
        1,
        "evidence_linked",
        EvidenceLinkedPayload(
            link_id="link-1",
            trajectory_id="traj-1",
            claim_id="claim-1",
            curated_item_id="cur-1",
            relationship="supports",
            strength=0.9,
        ),
        "idem-1",
    )
    snap2 = reduce_event(snap, event)
    assert len(snap2.evidence_links) == 1
    assert snap2.evidence_links[0].link_id == "link-1"
    assert snap2.evidence_links[0].relationship == "supports"
    assert snap2.evidence_links[0].strength == 0.9
    assert snap2.claims[0].evidence_link_ids == ("link-1",)


def test_evidence_linked_unknown_claim_fails() -> None:
    snap = _empty_snapshot()
    event = _event(
        snap,
        1,
        "evidence_linked",
        EvidenceLinkedPayload(
            link_id="link-1",
            trajectory_id="traj-1",
            claim_id="claim-1",
            curated_item_id="cur-1",
            relationship="supports",
            strength=0.9,
        ),
        "idem-1",
    )
    with pytest.raises(InvariantError, match="unknown claim"):
        reduce_event(snap, event)


def test_evidence_linked_unknown_curated_fails() -> None:
    snap = HarnessSnapshot.create(
        trajectory_id="traj-1",
        epoch_id="epoch-0",
        sequence=0,
        claims=(
            Claim.create(
                claim_id="claim-1",
                trajectory_id="traj-1",
                content="foo",
            ),
        ),
    )
    event = _event(
        snap,
        1,
        "evidence_linked",
        EvidenceLinkedPayload(
            link_id="link-1",
            trajectory_id="traj-1",
            claim_id="claim-1",
            curated_item_id="cur-1",
            relationship="supports",
            strength=0.9,
        ),
        "idem-1",
    )
    with pytest.raises(InvariantError, match="unknown curated item"):
        reduce_event(snap, event)


def test_duplicate_evidence_linked_fails() -> None:
    snap = HarnessSnapshot.create(
        trajectory_id="traj-1",
        epoch_id="epoch-0",
        sequence=0,
        claims=(
            Claim.create(
                claim_id="claim-1",
                trajectory_id="traj-1",
                content="foo",
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
        "evidence_linked",
        EvidenceLinkedPayload(
            link_id="link-1",
            trajectory_id="traj-1",
            claim_id="claim-1",
            curated_item_id="cur-1",
            relationship="supports",
            strength=0.9,
        ),
        "idem-1",
    )
    with pytest.raises(InvariantError, match="Duplicate EvidenceLink id"):
        reduce_event(snap, event)


def test_wrong_chain_hash_fails() -> None:
    snap = _empty_snapshot()
    first = _event(
        snap,
        1,
        "action_committed",
        ActionCommittedPayload(action_type="search", action_params=()),
        "idem-1",
    )
    snap2 = reduce_event(snap, first)
    bogus = HarnessEvent.create(
        event_id="evt-2",
        trajectory_id="traj-1",
        sequence=2,
        event_type="action_committed",
        payload=ActionCommittedPayload(action_type="read", action_params=()),
        idempotency_key="idem-2",
        previous_event_hash="c" * 64,
    )
    with pytest.raises(InvariantError, match="Event chain mismatch"):
        reduce_event(snap2, bogus)


def _signed_event(
    snapshot: HarnessSnapshot,
    sequence: int,
    event_type: str,
    payload: object,
    idempotency_key: str,
    signing_key: bytes,
    event_id: str | None = None,
) -> HarnessEvent:
    return HarnessEvent.create(
        event_id=event_id or f"evt-{sequence}",
        trajectory_id=snapshot.trajectory_id,
        sequence=sequence,
        event_type=event_type,
        payload=payload,  # type: ignore[arg-type]
        idempotency_key=idempotency_key,
        previous_event_hash=snapshot.last_event_hash,
        previous_signature=snapshot.last_signature,
        signing_key=signing_key,
        actor="policy",
        action_id="act-1",
    )


def test_reducer_tracks_signature_chain() -> None:
    key = b"reducer-key"
    snap = _empty_snapshot()
    first = _signed_event(
        snap, 1, "action_committed", ActionCommittedPayload("search", ()), "idem-1", key
    )
    snap1 = reduce_event(snap, first)
    assert snap1.last_signature == first.header.signature

    second = _signed_event(
        snap1, 2, "action_committed", ActionCommittedPayload("search", ()), "idem-2", key
    )
    assert second.header.previous_signature == first.header.signature
    snap2 = reduce_event(snap1, second)
    assert snap2.last_signature == second.header.signature


def test_reducer_rejects_broken_signature_chain() -> None:
    key = b"reducer-key"
    snap = _empty_snapshot()
    first = _signed_event(
        snap, 1, "action_committed", ActionCommittedPayload("search", ()), "idem-1", key
    )
    snap1 = reduce_event(snap, first)

    # Build a second event with a wrong previous signature.
    second = HarnessEvent.create(
        event_id="evt-2",
        trajectory_id=snap.trajectory_id,
        sequence=2,
        event_type="action_committed",
        payload=ActionCommittedPayload("search", ()),
        idempotency_key="idem-2",
        previous_event_hash=first.header.event_hash,
        previous_signature=ContentHash("a" * 64),
        signing_key=key,
    )
    with pytest.raises(InvariantError, match="Signature chain mismatch"):
        reduce_event(snap1, second)
