"""Append-only event schemas for the harness control plane."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

from rfsn_agent.common import canonical_json, hash_content, utc_now
from rfsn_agent.types import (
    ClaimId,
    ClaimStatus,
    ContentHash,
    EventId,
    ItemId,
    LinkId,
    SubmissionId,
    TaskId,
    ToolInvocationId,
    TrajectoryId,
    VerificationId,
    VerificationResult,
)

CURRENT_EVENT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class EventHeader:
    """Immutable envelope shared by every harness event."""

    event_id: EventId
    trajectory_id: TrajectoryId
    sequence: int
    event_type: str
    schema_version: int
    idempotency_key: str
    parent_event_id: EventId | None
    created_at: datetime
    actor: str
    action_id: str


# ---------------------------------------------------------------------------
# Typed event payloads
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ActionCommittedPayload:
    action_type: str
    action_params: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class ToolInvokedPayload:
    invocation_id: ToolInvocationId
    parent_task_id: TaskId | None
    tool_name: str
    arguments: tuple[tuple[str, str], ...]
    dependency_ids: tuple[ToolInvocationId, ...]
    deadline: datetime | None


@dataclass(frozen=True, slots=True)
class ToolResultReceivedPayload:
    invocation_id: ToolInvocationId
    status: str
    content: str


@dataclass(frozen=True, slots=True)
class EvidenceCuratedPayload:
    candidate_ids: tuple[ItemId, ...]
    curated_item_id: ItemId
    content: str
    priority: int
    source_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ClaimRevisedPayload:
    claim_id: ClaimId
    new_content: str | None
    new_status: ClaimStatus | None


@dataclass(frozen=True, slots=True)
class EvidenceVerifiedPayload:
    link_id: LinkId
    verification_id: VerificationId
    result: VerificationResult
    details: str


@dataclass(frozen=True, slots=True)
class TaskDecomposedPayload:
    parent_task_id: TaskId | None
    task_id: TaskId
    description: str
    dependency_ids: tuple[TaskId, ...]


@dataclass(frozen=True, slots=True)
class TaskCompletedPayload:
    task_id: TaskId


@dataclass(frozen=True, slots=True)
class ContextPrunedPayload:
    retained_item_ids: tuple[ItemId, ...]
    new_epoch_id: str


@dataclass(frozen=True, slots=True)
class SubmissionRecordedPayload:
    submission_id: SubmissionId
    content: str
    source_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SnapshotCheckpointedPayload:
    snapshot_sequence: int
    snapshot_hash: ContentHash


EventPayload = (
    ActionCommittedPayload
    | ToolInvokedPayload
    | ToolResultReceivedPayload
    | EvidenceCuratedPayload
    | ClaimRevisedPayload
    | EvidenceVerifiedPayload
    | TaskDecomposedPayload
    | TaskCompletedPayload
    | ContextPrunedPayload
    | SubmissionRecordedPayload
    | SnapshotCheckpointedPayload
)


# ---------------------------------------------------------------------------
# Event envelope
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HarnessEvent:
    """An append-only, content-addressed event in a trajectory."""

    header: EventHeader
    payload: EventPayload
    payload_hash: ContentHash

    def __post_init__(self) -> None:
        expected = hash_content(canonical_json(_payload_to_dict(self.payload)))
        if expected != self.payload_hash:
            raise ValueError(
                f"HarnessEvent {self.header.event_id}: payload_hash mismatch: "
                f"expected {expected}, got {self.payload_hash}"
            )

    @property
    def event_id(self) -> EventId:
        return self.header.event_id

    @property
    def trajectory_id(self) -> TrajectoryId:
        return self.header.trajectory_id

    @property
    def sequence(self) -> int:
        return self.header.sequence

    @property
    def event_type(self) -> str:
        return self.header.event_type

    @property
    def idempotency_key(self) -> str:
        return self.header.idempotency_key

    def payload_to_dict(self) -> dict[str, Any]:
        return _payload_to_dict(self.payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "header": {
                "event_id": self.header.event_id,
                "trajectory_id": self.header.trajectory_id,
                "sequence": self.header.sequence,
                "event_type": self.header.event_type,
                "schema_version": self.header.schema_version,
                "idempotency_key": self.header.idempotency_key,
                "parent_event_id": self.header.parent_event_id,
                "created_at": self.header.created_at.isoformat(),
                "actor": self.header.actor,
                "action_id": self.header.action_id,
            },
            "payload": self.payload_to_dict(),
            "payload_hash": self.payload_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HarnessEvent:
        """Reconstruct a validated HarnessEvent from its serialized form."""
        header = _header_from_dict(data["header"])
        payload = _payload_from_dict(header.event_type, data["payload"])
        return cls(
            header=header,
            payload=payload,
            payload_hash=ContentHash(data["payload_hash"]),
        )

    @classmethod
    def create(
        cls,
        *,
        event_id: EventId,
        trajectory_id: TrajectoryId,
        sequence: int,
        event_type: str,
        payload: EventPayload,
        idempotency_key: str,
        parent_event_id: EventId | None = None,
        actor: str = "system",
        action_id: str = "unknown",
    ) -> HarnessEvent:
        header = EventHeader(
            event_id=event_id,
            trajectory_id=trajectory_id,
            sequence=sequence,
            event_type=event_type,
            schema_version=CURRENT_EVENT_SCHEMA_VERSION,
            idempotency_key=idempotency_key,
            parent_event_id=parent_event_id,
            created_at=utc_now(),
            actor=actor,
            action_id=action_id,
        )
        payload_hash = hash_content(canonical_json(_payload_to_dict(payload)))
        return cls(header=header, payload=payload, payload_hash=payload_hash)


def _payload_to_dict(payload: EventPayload) -> dict[str, Any]:
    """Serialize a typed event payload into a deterministic dictionary."""

    def convert(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, bool | int | float | str):
            return value
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, ClaimStatus | VerificationResult):
            return value.value
        if isinstance(value, tuple | list):
            return [convert(v) for v in value]
        if hasattr(value, "__dataclass_fields__"):
            return {
                field_name: convert(getattr(value, field_name))
                for field_name in value.__dataclass_fields__
            }
        return str(value)

    return cast(dict[str, Any], convert(payload))


def payload_type_name(payload: EventPayload) -> str:
    """Return the canonical event type string for a payload instance."""
    mapping: dict[type[EventPayload], str] = {
        ActionCommittedPayload: "action_committed",
        ToolInvokedPayload: "tool_invoked",
        ToolResultReceivedPayload: "tool_result_received",
        EvidenceCuratedPayload: "evidence_curated",
        ClaimRevisedPayload: "claim_revised",
        EvidenceVerifiedPayload: "evidence_verified",
        TaskDecomposedPayload: "task_decomposed",
        TaskCompletedPayload: "task_completed",
        ContextPrunedPayload: "context_pruned",
        SubmissionRecordedPayload: "submission_recorded",
        SnapshotCheckpointedPayload: "snapshot_checkpointed",
    }
    return mapping[type(payload)]


# ---------------------------------------------------------------------------
# Deserialization helpers
# ---------------------------------------------------------------------------


def _parse_datetime(value: str) -> datetime:
    """Parse an ISO-format datetime string produced by canonical serialization."""
    return datetime.fromisoformat(value)


def _parse_optional_datetime(value: str | None) -> datetime | None:
    return _parse_datetime(value) if value is not None else None


def _header_from_dict(data: dict[str, Any]) -> EventHeader:
    return EventHeader(
        event_id=EventId(data["event_id"]),
        trajectory_id=TrajectoryId(data["trajectory_id"]),
        sequence=int(data["sequence"]),
        event_type=str(data["event_type"]),
        schema_version=int(data["schema_version"]),
        idempotency_key=str(data["idempotency_key"]),
        parent_event_id=EventId(data["parent_event_id"])
        if data.get("parent_event_id") is not None
        else None,
        created_at=_parse_datetime(data["created_at"]),
        actor=str(data["actor"]),
        action_id=str(data["action_id"]),
    )


def _payload_from_dict(event_type: str, data: dict[str, Any]) -> EventPayload:
    """Reconstruct a typed payload from its serialized dictionary form."""
    if event_type == "action_committed":
        return ActionCommittedPayload(
            action_type=data["action_type"],
            action_params=tuple(tuple(p) for p in data["action_params"]),
        )
    if event_type == "tool_invoked":
        return ToolInvokedPayload(
            invocation_id=ToolInvocationId(data["invocation_id"]),
            parent_task_id=TaskId(data["parent_task_id"])
            if data.get("parent_task_id") is not None
            else None,
            tool_name=data["tool_name"],
            arguments=tuple(tuple(a) for a in data["arguments"]),
            dependency_ids=tuple(ToolInvocationId(d) for d in data["dependency_ids"]),
            deadline=_parse_optional_datetime(data.get("deadline")),
        )
    if event_type == "tool_result_received":
        return ToolResultReceivedPayload(
            invocation_id=ToolInvocationId(data["invocation_id"]),
            status=data["status"],
            content=data["content"],
        )
    if event_type == "evidence_curated":
        return EvidenceCuratedPayload(
            candidate_ids=tuple(ItemId(cid) for cid in data["candidate_ids"]),
            curated_item_id=ItemId(data["curated_item_id"]),
            content=data["content"],
            priority=int(data["priority"]),
            source_ids=tuple(data["source_ids"]),
        )
    if event_type == "claim_revised":
        return ClaimRevisedPayload(
            claim_id=ClaimId(data["claim_id"]),
            new_content=data.get("new_content"),
            new_status=ClaimStatus(data["new_status"])
            if data.get("new_status") is not None
            else None,
        )
    if event_type == "evidence_verified":
        return EvidenceVerifiedPayload(
            link_id=LinkId(data["link_id"]),
            verification_id=VerificationId(data["verification_id"]),
            result=VerificationResult(data["result"]),
            details=data["details"],
        )
    if event_type == "task_decomposed":
        return TaskDecomposedPayload(
            parent_task_id=TaskId(data["parent_task_id"])
            if data.get("parent_task_id") is not None
            else None,
            task_id=TaskId(data["task_id"]),
            description=data["description"],
            dependency_ids=tuple(TaskId(d) for d in data["dependency_ids"]),
        )
    if event_type == "task_completed":
        return TaskCompletedPayload(task_id=TaskId(data["task_id"]))
    if event_type == "context_pruned":
        return ContextPrunedPayload(
            retained_item_ids=tuple(ItemId(i) for i in data["retained_item_ids"]),
            new_epoch_id=data["new_epoch_id"],
        )
    if event_type == "submission_recorded":
        return SubmissionRecordedPayload(
            submission_id=SubmissionId(data["submission_id"]),
            content=data["content"],
            source_ids=tuple(data["source_ids"]),
        )
    if event_type == "snapshot_checkpointed":
        return SnapshotCheckpointedPayload(
            snapshot_sequence=int(data["snapshot_sequence"]),
            snapshot_hash=ContentHash(data["snapshot_hash"]),
        )
    raise ValueError(f"Cannot deserialize unknown event type: {event_type}")
