"""Tests for event schemas."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime

import pytest

from rfsn_agent.common import canonical_json, hash_content
from rfsn_agent.events import (
    CURRENT_EVENT_SCHEMA_VERSION,
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
    compute_event_hash,
    payload_type_name,
)
from rfsn_agent.types import ClaimStatus, VerificationResult


def test_event_create_and_immutable() -> None:
    payload = ActionCommittedPayload(
        action_type="search",
        action_params=(("query", "foo"),),
    )
    event = HarnessEvent.create(
        event_id="evt-1",
        trajectory_id="traj-1",
        sequence=0,
        event_type="action_committed",
        payload=payload,
        idempotency_key="idem-1",
        previous_event_hash=None,
        actor="policy",
        action_id="act-1",
    )
    assert event.trajectory_id == "traj-1"
    assert event.sequence == 0
    assert event.header.schema_version == CURRENT_EVENT_SCHEMA_VERSION
    assert event.payload_hash == hash_content(canonical_json(event.payload_to_dict()))

    with pytest.raises(dataclasses.FrozenInstanceError):
        event.header.sequence = 99  # type: ignore[misc]


def test_event_payload_hash_validation() -> None:
    payload = ActionCommittedPayload(action_type="search", action_params=())
    bad_hash = "badhash"
    header = HarnessEvent.create(
        event_id="evt-1",
        trajectory_id="traj-1",
        sequence=0,
        event_type="action_committed",
        payload=payload,
        idempotency_key="idem-1",
        previous_event_hash=None,
    ).header
    with pytest.raises(ValueError, match="payload_hash mismatch"):
        HarnessEvent(header=header, payload=payload, payload_hash=bad_hash)


def test_event_hash_covers_full_envelope() -> None:
    payload = ActionCommittedPayload(action_type="search", action_params=())
    event = HarnessEvent.create(
        event_id="evt-1",
        trajectory_id="traj-1",
        sequence=1,
        event_type="action_committed",
        payload=payload,
        idempotency_key="idem-1",
        previous_event_hash=None,
        actor="policy",
        action_id="act-1",
    )

    # Mutating any envelope field should invalidate the event hash.
    for field_name in (
        "event_id",
        "trajectory_id",
        "sequence",
        "event_type",
        "schema_version",
        "idempotency_key",
        "parent_event_id",
        "actor",
        "action_id",
        "previous_event_hash",
    ):
        header_dict = event.header.__dict__ if hasattr(event.header, "__dict__") else {
            f: getattr(event.header, f)
            for f in event.header.__dataclass_fields__
        }
        mutated = dict(header_dict)
        if field_name == "parent_event_id":
            mutated[field_name] = "evt-parent"
        elif field_name == "previous_event_hash":
            mutated[field_name] = "a" * 64
        elif field_name == "sequence":
            mutated[field_name] = 99
        elif field_name == "schema_version":
            mutated[field_name] = 999
        else:
            mutated[field_name] = "mutated"
        new_header = event.header.__class__(**mutated)
        with pytest.raises(ValueError, match="event_hash mismatch"):
            HarnessEvent(header=new_header, payload=payload, payload_hash=event.payload_hash)


def test_event_hash_is_canonical() -> None:
    """Different field partitions must not collapse to the same hash input."""
    payload = ActionCommittedPayload(action_type="search", action_params=())
    e1 = HarnessEvent.create(
        event_id="evt-1",
        trajectory_id="traj-12",
        sequence=3,
        event_type="action_committed",
        payload=payload,
        idempotency_key="idem-1",
        previous_event_hash=None,
    )
    e2 = HarnessEvent.create(
        event_id="evt-112",
        trajectory_id="traj-3",
        sequence=3,
        event_type="action_committed",
        payload=payload,
        idempotency_key="idem-1",
        previous_event_hash=None,
    )
    assert e1.header.event_hash != e2.header.event_hash


def test_tool_invoked_payload_serialization() -> None:
    deadline = datetime(2026, 1, 1, tzinfo=UTC)
    payload = ToolInvokedPayload(
        invocation_id="tool-1",
        parent_task_id="task-1",
        tool_name="read_file",
        arguments=(("path", "/tmp/foo"),),
        dependency_ids=("tool-0",),
        deadline=deadline,
    )
    d = HarnessEvent.create(
        event_id="evt-1",
        trajectory_id="traj-1",
        sequence=0,
        event_type="tool_invoked",
        payload=payload,
        idempotency_key="idem-1",
        previous_event_hash=None,
    ).payload_to_dict()
    assert d["invocation_id"] == "tool-1"
    assert d["deadline"] == "2026-01-01T00:00:00+00:00"


def test_claim_revised_payload_enum_serialization() -> None:
    payload = ClaimRevisedPayload(
        claim_id="claim-1",
        new_content=None,
        new_status=ClaimStatus.VERIFIED,
    )
    d = HarnessEvent.create(
        event_id="evt-1",
        trajectory_id="traj-1",
        sequence=0,
        event_type="claim_revised",
        payload=payload,
        idempotency_key="idem-1",
        previous_event_hash=None,
    ).payload_to_dict()
    assert d["new_status"] == "verified"


def test_all_payload_round_trip_through_dict() -> None:
    """Every payload must serialize deterministically and validate its hash."""
    payloads: list[object] = [
        ActionCommittedPayload(action_type="search", action_params=(("q", "x"),)),
        ToolInvokedPayload(
            invocation_id="t1",
            parent_task_id=None,
            tool_name="read",
            arguments=(),
            dependency_ids=(),
            deadline=None,
        ),
        ToolResultReceivedPayload(invocation_id="t1", status="success", content="ok"),
        EvidenceCuratedPayload(
            candidate_ids=("c1",),
            curated_item_id="cur-1",
            content="curated",
            priority=3,
            source_ids=("src-1",),
        ),
        ClaimRevisedPayload(
            claim_id="claim-1", new_content="updated", new_status=ClaimStatus.CONTRADICTED
        ),
        EvidenceVerifiedPayload(
            link_id="link-1",
            verification_id="ver-1",
            result=VerificationResult.CONFIRMED,
            details="two sources agree",
        ),
        TaskDecomposedPayload(
            parent_task_id=None,
            task_id="task-1",
            description="do it",
            dependency_ids=(),
        ),
        TaskCompletedPayload(task_id="task-1"),
        ContextPrunedPayload(retained_item_ids=("cur-1",), new_epoch_id="epoch-2"),
        SubmissionRecordedPayload(
            submission_id="sub-1", content="answer", source_ids=("src-1",)
        ),
        SnapshotCheckpointedPayload(snapshot_sequence=5, snapshot_hash="a" * 64),
        CandidateAddedPayload(
            item_id="cand-1",
            trajectory_id="traj-1",
            source_id="src-1",
            retrieval_query="q",
            content="candidate body",
        ),
        ClaimCreatedPayload(
            claim_id="claim-1",
            trajectory_id="traj-1",
            content="claim body",
        ),
        EvidenceLinkedPayload(
            link_id="link-1",
            trajectory_id="traj-1",
            claim_id="claim-1",
            curated_item_id="cur-1",
            relationship="supports",
            strength=0.9,
        ),
    ]
    for idx, payload in enumerate(payloads):
        event_type = payload_type_name(payload)  # type: ignore[arg-type]
        previous_hash = None if idx == 0 else "a" * 64
        event = HarnessEvent.create(
            event_id=f"evt-{idx}",
            trajectory_id="traj-1",
            sequence=idx,
            event_type=event_type,
            payload=payload,  # type: ignore[arg-type]
            idempotency_key=f"idem-{idx}",
            previous_event_hash=previous_hash,
        )
        assert event.payload_hash == hash_content(canonical_json(event.payload_to_dict()))


def test_event_hash_excludes_only_itself() -> None:
    """The event hash must include the payload hash and every header field."""
    payload = ActionCommittedPayload(action_type="search", action_params=())
    header = HarnessEvent.create(
        event_id="evt-1",
        trajectory_id="traj-1",
        sequence=1,
        event_type="action_committed",
        payload=payload,
        idempotency_key="idem-1",
        previous_event_hash=None,
    ).header
    expected = compute_event_hash(header, hash_content(canonical_json(payload)))
    assert header.event_hash == expected


def test_event_create_with_signing_key_includes_signature() -> None:
    payload = ActionCommittedPayload(action_type="search", action_params=())
    key = b"super-secret-key"
    event = HarnessEvent.create(
        event_id="evt-1",
        trajectory_id="traj-1",
        sequence=1,
        event_type="action_committed",
        payload=payload,
        idempotency_key="idem-1",
        previous_event_hash=None,
        signing_key=key,
    )
    assert event.header.signature is not None
    assert len(event.header.signature) == 64


def test_event_signature_chain_links_previous_signature() -> None:
    key = b"super-secret-key"
    payload = ActionCommittedPayload(action_type="search", action_params=())
    first = HarnessEvent.create(
        event_id="evt-1",
        trajectory_id="traj-1",
        sequence=1,
        event_type="action_committed",
        payload=payload,
        idempotency_key="idem-1",
        previous_event_hash=None,
        signing_key=key,
    )
    second = HarnessEvent.create(
        event_id="evt-2",
        trajectory_id="traj-1",
        sequence=2,
        event_type="action_committed",
        payload=payload,
        idempotency_key="idem-2",
        previous_event_hash=first.header.event_hash,
        previous_signature=first.header.signature,
        signing_key=key,
    )
    assert second.header.previous_signature == first.header.signature
    assert second.header.signature != first.header.signature


def test_event_signature_verification() -> None:
    from rfsn_agent.events import verify_signature

    payload = ActionCommittedPayload(action_type="search", action_params=())
    key = b"super-secret-key"
    event = HarnessEvent.create(
        event_id="evt-1",
        trajectory_id="traj-1",
        sequence=1,
        event_type="action_committed",
        payload=payload,
        idempotency_key="idem-1",
        previous_event_hash=None,
        signing_key=key,
    )
    verify_signature(event.header, key)

    with pytest.raises(ValueError, match="signature mismatch"):
        verify_signature(event.header, b"wrong-key")


def test_unsigned_event_rejects_verification_with_key() -> None:
    from rfsn_agent.events import verify_signature

    payload = ActionCommittedPayload(action_type="search", action_params=())
    event = HarnessEvent.create(
        event_id="evt-1",
        trajectory_id="traj-1",
        sequence=1,
        event_type="action_committed",
        payload=payload,
        idempotency_key="idem-1",
        previous_event_hash=None,
    )
    verify_signature(event.header, None)
    with pytest.raises(ValueError, match="missing signature"):
        verify_signature(event.header, b"some-key")
