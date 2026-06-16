"""Tests for event schemas."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime

import pytest

from rfsn_agent.common import canonical_json, hash_content
from rfsn_agent.events import (
    CURRENT_EVENT_SCHEMA_VERSION,
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
    ).header
    with pytest.raises(ValueError, match="payload_hash mismatch"):
        HarnessEvent(header=header, payload=payload, payload_hash=bad_hash)


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
    ]
    for idx, payload in enumerate(payloads):
        event_type = payload_type_name(payload)  # type: ignore[arg-type]
        event = HarnessEvent.create(
            event_id=f"evt-{idx}",
            trajectory_id="traj-1",
            sequence=idx,
            event_type=event_type,
            payload=payload,  # type: ignore[arg-type]
            idempotency_key=f"idem-{idx}",
        )
        assert event.payload_hash == hash_content(canonical_json(event.payload_to_dict()))
