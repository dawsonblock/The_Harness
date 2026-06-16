"""Pure event reducer: previous snapshot + validated event -> new snapshot."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from rfsn_agent.domain import (
    BudgetLedger,
    CandidateItem,
    Claim,
    CuratedItem,
    EvidenceLink,
    HarnessSnapshot,
    Provenance,
    SubmissionRecord,
    TaskNode,
    ToolInvocation,
    ToolResult,
    VerificationRecord,
)
from rfsn_agent.events import (
    CURRENT_EVENT_SCHEMA_VERSION,
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
from rfsn_agent.types import TaskStatus, ToolStatus, VerificationResult, VerificationStatus


class ReducerError(ValueError):
    """Raised when an event cannot be reduced into a new snapshot."""


class InvariantError(ReducerError):
    """Raised when applying an event would violate a state invariant."""


EventHandler = Callable[[HarnessSnapshot, HarnessEvent], HarnessSnapshot]


def reduce_event(snapshot: HarnessSnapshot, event: HarnessEvent) -> HarnessSnapshot:
    """Return the next snapshot produced by applying ``event`` to ``snapshot``.

    The reducer is pure and deterministic: the same snapshot and event always
    produce the same resulting snapshot. It validates envelope invariants and
    event-specific preconditions before mutating state.
    """
    _validate_envelope(snapshot, event)

    handler = _HANDLERS.get(event.event_type)
    if handler is None:
        raise ReducerError(f"Unknown event type: {event.event_type}")

    return handler(snapshot, event)


def _validate_envelope(snapshot: HarnessSnapshot, event: HarnessEvent) -> None:
    if event.trajectory_id != snapshot.trajectory_id:
        raise InvariantError(
            f"Trajectory mismatch: event {event.event_id} belongs to "
            f"{event.trajectory_id}, snapshot is {snapshot.trajectory_id}"
        )

    if event.header.schema_version > CURRENT_EVENT_SCHEMA_VERSION:
        raise InvariantError(
            f"Unsupported event schema version: {event.header.schema_version} "
            f"(current {CURRENT_EVENT_SCHEMA_VERSION})"
        )

    expected_type = payload_type_name(event.payload)
    if event.header.event_type != expected_type:
        raise InvariantError(
            f"Event type mismatch: header says {event.header.event_type!r}, "
            f"payload is {expected_type!r}"
        )

    if event.sequence != snapshot.sequence + 1:
        raise InvariantError(
            f"Sequence gap: snapshot={snapshot.sequence}, event={event.sequence}"
        )

    if event.sequence == 1 and event.header.previous_event_hash is not None:
        raise InvariantError("First event must not have a previous hash")

    if event.sequence > 1 and event.header.previous_event_hash is None:
        raise InvariantError("Non-genesis event requires a previous hash")

    if event.header.previous_event_hash != snapshot.last_event_hash:
        raise InvariantError(
            "Event chain mismatch: expected previous hash "
            f"{snapshot.last_event_hash}, received {event.header.previous_event_hash}"
        )

    # Signature chain validation. When any event in the trajectory is signed,
    # the entire chain must be signed consistently.
    if event.header.signature is not None or snapshot.last_signature is not None:
        if event.sequence == 1 and event.header.previous_signature is not None:
            raise InvariantError("First event must not have a previous signature")
        if event.sequence > 1 and event.header.previous_signature is None:
            raise InvariantError("Non-genesis event requires a previous signature")
        if event.header.previous_signature != snapshot.last_signature:
            raise InvariantError(
                "Signature chain mismatch: expected previous signature "
                f"{snapshot.last_signature}, received {event.header.previous_signature}"
            )


def _with_event_provenance(event: HarnessEvent) -> Provenance:
    return Provenance(
        actor=event.header.actor,
        action_id=event.header.action_id,
        event_id=event.event_id,
    )


def _next_snapshot(
    snapshot: HarnessSnapshot,
    event: HarnessEvent,
    *,
    epoch_id: str | None = None,
    candidates: tuple[CandidateItem, ...] | None = None,
    curated_items: tuple[CuratedItem, ...] | None = None,
    claims: tuple[Claim, ...] | None = None,
    evidence_links: tuple[EvidenceLink, ...] | None = None,
    verification_records: tuple[VerificationRecord, ...] | None = None,
    tasks: tuple[TaskNode, ...] | None = None,
    budget: BudgetLedger | None = None,
    submissions: tuple[SubmissionRecord, ...] | None = None,
    tool_invocations: tuple[ToolInvocation, ...] | None = None,
    tool_results: tuple[ToolResult, ...] | None = None,
) -> HarnessSnapshot:
    """Build a new snapshot with updated sequence and chain hash."""
    return HarnessSnapshot.create(
        trajectory_id=snapshot.trajectory_id,
        epoch_id=epoch_id if epoch_id is not None else snapshot.epoch_id,
        sequence=event.sequence,
        last_event_hash=event.header.event_hash,
        last_signature=event.header.signature,
        created_at=event.header.created_at,
        candidates=candidates if candidates is not None else snapshot.candidates,
        curated_items=curated_items if curated_items is not None else snapshot.curated_items,
        claims=claims if claims is not None else snapshot.claims,
        evidence_links=evidence_links if evidence_links is not None else snapshot.evidence_links,
        verification_records=verification_records
        if verification_records is not None
        else snapshot.verification_records,
        tasks=tasks if tasks is not None else snapshot.tasks,
        budget=budget if budget is not None else snapshot.budget,
        submissions=submissions if submissions is not None else snapshot.submissions,
        tool_invocations=tool_invocations
        if tool_invocations is not None
        else snapshot.tool_invocations,
        tool_results=tool_results if tool_results is not None else snapshot.tool_results,
    )


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------


def _handle_action_committed(
    snapshot: HarnessSnapshot, event: HarnessEvent
) -> HarnessSnapshot:
    # ActionCommitted is primarily an audit event; semantic state changes are
    # carried by subsequent specialized events.
    return _next_snapshot(snapshot, event)


def _handle_tool_invoked(snapshot: HarnessSnapshot, event: HarnessEvent) -> HarnessSnapshot:
    payload = _expect_payload_type(event.payload, ToolInvokedPayload)
    _require_unique(
        payload.invocation_id,
        {t.invocation_id for t in snapshot.tool_invocations},
        "ToolInvocation",
    )
    invocation = ToolInvocation.create(
        invocation_id=payload.invocation_id,
        trajectory_id=event.trajectory_id,
        action_id=event.header.action_id,
        parent_task_id=payload.parent_task_id,
        tool_name=payload.tool_name,
        arguments=payload.arguments,
        dependency_ids=payload.dependency_ids,
        deadline=payload.deadline,
        provenance=_with_event_provenance(event),
    )
    budget = snapshot.budget.spend(tokens=0, tool_calls=1) if snapshot.budget is not None else None
    return _next_snapshot(
        snapshot,
        event,
        budget=budget,
        tool_invocations=snapshot.tool_invocations + (invocation,),
    )


def _handle_tool_result_received(
    snapshot: HarnessSnapshot, event: HarnessEvent
) -> HarnessSnapshot:
    payload = _expect_payload_type(event.payload, ToolResultReceivedPayload)
    invocation_ids = {t.invocation_id for t in snapshot.tool_invocations}
    if payload.invocation_id not in invocation_ids:
        raise InvariantError(
            f"ToolResult references unknown invocation: {payload.invocation_id}"
        )
    result_ids = {r.invocation_id for r in snapshot.tool_results}
    if payload.invocation_id in result_ids:
        raise InvariantError(
            f"Duplicate ToolResult for invocation: {payload.invocation_id}"
        )
    status = ToolStatus(payload.status)
    result = ToolResult.create(
        result_id=payload.invocation_id,
        invocation_id=payload.invocation_id,
        trajectory_id=event.trajectory_id,
        status=status,
        content=payload.content,
        provenance=_with_event_provenance(event),
    )
    return _next_snapshot(
        snapshot,
        event,
        tool_results=snapshot.tool_results + (result,),
    )


def _handle_evidence_curated(
    snapshot: HarnessSnapshot, event: HarnessEvent
) -> HarnessSnapshot:
    payload = _expect_payload_type(event.payload, EvidenceCuratedPayload)
    candidate_ids = {c.item_id for c in snapshot.candidates}
    missing = [cid for cid in payload.candidate_ids if cid not in candidate_ids]
    if missing:
        raise InvariantError(f"Evidence curation references unknown candidates: {missing}")
    _require_unique(
        payload.curated_item_id,
        {c.item_id for c in snapshot.curated_items},
        "CuratedItem",
    )
    curated = CuratedItem.create(
        item_id=payload.curated_item_id,
        trajectory_id=event.trajectory_id,
        candidate_ids=payload.candidate_ids,
        content=payload.content,
        priority=payload.priority,
        source_ids=payload.source_ids,
        provenance=_with_event_provenance(event),
    )
    return _next_snapshot(
        snapshot,
        event,
        curated_items=snapshot.curated_items + (curated,),
    )


def _handle_claim_revised(
    snapshot: HarnessSnapshot, event: HarnessEvent
) -> HarnessSnapshot:
    payload = _expect_payload_type(event.payload, ClaimRevisedPayload)
    claim_index = _index_by_id(snapshot.claims, lambda c: c.claim_id, payload.claim_id)
    old_claim = snapshot.claims[claim_index]
    new_content = payload.new_content if payload.new_content is not None else old_claim.content
    new_status = payload.new_status if payload.new_status is not None else old_claim.status
    revised = Claim.create(
        claim_id=old_claim.claim_id,
        trajectory_id=old_claim.trajectory_id,
        content=new_content,
        status=new_status,
        evidence_link_ids=old_claim.evidence_link_ids,
        provenance=_with_event_provenance(event),
    )
    claims = _replace_at(snapshot.claims, claim_index, revised)
    return _next_snapshot(snapshot, event, claims=claims)


def _handle_evidence_verified(
    snapshot: HarnessSnapshot, event: HarnessEvent
) -> HarnessSnapshot:
    payload = _expect_payload_type(event.payload, EvidenceVerifiedPayload)
    link_index = _index_by_id(
        snapshot.evidence_links, lambda link: link.link_id, payload.link_id
    )
    old_link = snapshot.evidence_links[link_index]
    if old_link.trajectory_id != event.trajectory_id:
        raise InvariantError(
            f"Evidence link {payload.link_id} does not belong to trajectory {event.trajectory_id}"
        )

    record_ids = {r.record_id for r in snapshot.verification_records}
    if payload.verification_id in record_ids:
        raise InvariantError(f"Duplicate verification record id: {payload.verification_id}")

    record = VerificationRecord.create(
        record_id=payload.verification_id,
        trajectory_id=event.trajectory_id,
        link_id=payload.link_id,
        claim_id=old_link.claim_id,
        method="event",
        result=payload.result,
        details=payload.details,
        provenance=_with_event_provenance(event),
    )

    current_status = _verification_status_from_result(payload.result)
    updated_link = EvidenceLink(
        link_id=old_link.link_id,
        trajectory_id=old_link.trajectory_id,
        claim_id=old_link.claim_id,
        curated_item_id=old_link.curated_item_id,
        relationship=old_link.relationship,
        strength=old_link.strength,
        current_status=current_status,
        verification_id=payload.verification_id,
        created_at=old_link.created_at,
        provenance=old_link.provenance,
    )
    evidence_links = _replace_at(snapshot.evidence_links, link_index, updated_link)
    return _next_snapshot(
        snapshot,
        event,
        evidence_links=evidence_links,
        verification_records=snapshot.verification_records + (record,),
    )


def _verification_status_from_result(result: VerificationResult) -> VerificationStatus:
    if result == VerificationResult.CONFIRMED:
        return VerificationStatus.VERIFIED
    if result == VerificationResult.REFUTED:
        return VerificationStatus.REFUTED
    return VerificationStatus.INCONCLUSIVE


def _handle_task_decomposed(
    snapshot: HarnessSnapshot, event: HarnessEvent
) -> HarnessSnapshot:
    payload = _expect_payload_type(event.payload, TaskDecomposedPayload)
    _require_unique(payload.task_id, {t.task_id for t in snapshot.tasks}, "TaskNode")
    task = TaskNode(
        task_id=payload.task_id,
        trajectory_id=event.trajectory_id,
        parent_id=payload.parent_task_id,
        description=payload.description,
        status=TaskStatus.PENDING,
        dependency_ids=payload.dependency_ids,
        created_at=event.header.created_at,
        provenance=_with_event_provenance(event),
    )
    return _next_snapshot(
        snapshot,
        event,
        tasks=snapshot.tasks + (task,),
    )


def _handle_task_completed(
    snapshot: HarnessSnapshot, event: HarnessEvent
) -> HarnessSnapshot:
    payload = _expect_payload_type(event.payload, TaskCompletedPayload)
    task_index = _index_by_id(snapshot.tasks, lambda t: t.task_id, payload.task_id)
    old_task = snapshot.tasks[task_index]
    if old_task.status in (TaskStatus.COMPLETED, TaskStatus.CANCELLED):
        raise InvariantError(
            f"Task {payload.task_id} is already {old_task.status.value}"
        )
    completed = TaskNode(
        task_id=old_task.task_id,
        trajectory_id=old_task.trajectory_id,
        parent_id=old_task.parent_id,
        description=old_task.description,
        status=TaskStatus.COMPLETED,
        dependency_ids=old_task.dependency_ids,
        created_at=old_task.created_at,
        completed_at=event.header.created_at,
        provenance=old_task.provenance,
    )
    tasks = _replace_at(snapshot.tasks, task_index, completed)
    return _next_snapshot(snapshot, event, tasks=tasks)


def _handle_context_pruned(
    snapshot: HarnessSnapshot, event: HarnessEvent
) -> HarnessSnapshot:
    payload = _expect_payload_type(event.payload, ContextPrunedPayload)
    retained_ids = set(payload.retained_item_ids)
    retained = tuple(c for c in snapshot.curated_items if c.item_id in retained_ids)
    unknown = retained_ids - {c.item_id for c in snapshot.curated_items}
    if unknown:
        raise InvariantError(
            f"Prune retained references unknown curated items: {sorted(unknown)}"
        )
    return _next_snapshot(
        snapshot,
        event,
        epoch_id=payload.new_epoch_id,
        curated_items=retained,
    )


def _handle_submission_recorded(
    snapshot: HarnessSnapshot, event: HarnessEvent
) -> HarnessSnapshot:
    payload = _expect_payload_type(event.payload, SubmissionRecordedPayload)
    _require_unique(
        payload.submission_id,
        {s.submission_id for s in snapshot.submissions},
        "SubmissionRecord",
    )
    submission = SubmissionRecord.create(
        submission_id=payload.submission_id,
        trajectory_id=event.trajectory_id,
        content=payload.content,
        source_ids=payload.source_ids,
        provenance=_with_event_provenance(event),
    )
    return _next_snapshot(
        snapshot,
        event,
        submissions=snapshot.submissions + (submission,),
    )


def _handle_snapshot_checkpointed(
    snapshot: HarnessSnapshot, event: HarnessEvent
) -> HarnessSnapshot:
    payload = _expect_payload_type(event.payload, SnapshotCheckpointedPayload)
    if payload.snapshot_sequence != snapshot.sequence:
        raise InvariantError(
            f"Checkpoint sequence mismatch: event={payload.snapshot_sequence}, "
            f"current={snapshot.sequence}"
        )
    current_hash = snapshot.compute_state_hash()
    if payload.snapshot_hash != current_hash:
        raise InvariantError(
            f"Checkpoint hash mismatch: event={payload.snapshot_hash}, "
            f"current={current_hash}"
        )
    return _next_snapshot(snapshot, event)


_HANDLERS: dict[str, EventHandler] = {
    "action_committed": _handle_action_committed,
    "tool_invoked": _handle_tool_invoked,
    "tool_result_received": _handle_tool_result_received,
    "evidence_curated": _handle_evidence_curated,
    "claim_revised": _handle_claim_revised,
    "evidence_verified": _handle_evidence_verified,
    "task_decomposed": _handle_task_decomposed,
    "task_completed": _handle_task_completed,
    "context_pruned": _handle_context_pruned,
    "submission_recorded": _handle_submission_recorded,
    "snapshot_checkpointed": _handle_snapshot_checkpointed,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


T = TypeVar("T")


def _expect_payload_type(payload: object, expected_type: type[T]) -> T:
    if not isinstance(payload, expected_type):
        raise ReducerError(
            f"Expected payload type {expected_type.__name__}, got {type(payload).__name__}"
        )
    return payload


def _require_unique(new_id: str, existing: set[str], kind: str) -> None:
    if new_id in existing:
        raise InvariantError(f"Duplicate {kind} id: {new_id}")


def _index_by_id(
    items: tuple[T, ...],
    id_accessor: Callable[[T], str],
    target_id: str,
) -> int:
    for idx, item in enumerate(items):
        if id_accessor(item) == target_id:
            return idx
    raise InvariantError(f"Unknown id: {target_id}")


def _replace_at(items: tuple[T, ...], index: int, new_item: T) -> tuple[T, ...]:
    return items[:index] + (new_item,) + items[index + 1 :]
