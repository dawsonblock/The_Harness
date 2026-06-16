"""Typed semantic actions, validation, and action-to-event conversion."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rfsn_agent.domain import HarnessSnapshot
from rfsn_agent.events import (
    ActionCommittedPayload,
    ClaimRevisedPayload,
    ContextPrunedPayload,
    EvidenceCuratedPayload,
    EvidenceVerifiedPayload,
    HarnessEvent,
    SubmissionRecordedPayload,
    TaskDecomposedPayload,
    ToolInvokedPayload,
)
from rfsn_agent.types import (
    ClaimId,
    ClaimStatus,
    EventId,
    ItemId,
    LinkId,
    SubmissionId,
    TaskId,
    ToolInvocationId,
    VerificationId,
    VerificationResult,
)


class ActionError(ValueError):
    """Raised when an action fails validation or cannot be applied."""


class SafetyError(ActionError):
    """Raised when an action violates the safety policy."""


class BudgetError(ActionError):
    """Raised when an action would exceed the remaining budget."""


class PreconditionError(ActionError):
    """Raised when an action's state preconditions are not met."""


# ---------------------------------------------------------------------------
# Action dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DecomposeAction:
    task_id: str
    parent_task_id: str | None
    description: str
    dependency_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SearchAction:
    query: str
    source_filter: str | None = None


@dataclass(frozen=True, slots=True)
class ReadAction:
    source_id: str
    query: str | None = None


@dataclass(frozen=True, slots=True)
class CurateAction:
    candidate_ids: tuple[str, ...]
    curated_item_id: str
    content: str
    priority: int
    source_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DiscardAction:
    candidate_ids: tuple[str, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class VerifyAction:
    link_id: str
    verification_id: str
    result: VerificationResult
    details: str


@dataclass(frozen=True, slots=True)
class ReviseClaimAction:
    claim_id: str
    new_content: str | None
    new_status: ClaimStatus | None


@dataclass(frozen=True, slots=True)
class PruneSemanticAction:
    retained_item_ids: tuple[str, ...]
    new_epoch_id: str


@dataclass(frozen=True, slots=True)
class RequestContextAction:
    context_type: str
    parameters: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class SubmitAction:
    submission_id: str
    content: str
    source_ids: tuple[str, ...]


Action = (
    DecomposeAction
    | SearchAction
    | ReadAction
    | CurateAction
    | DiscardAction
    | VerifyAction
    | ReviseClaimAction
    | PruneSemanticAction
    | RequestContextAction
    | SubmitAction
)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_action(
    action: Action,
    snapshot: HarnessSnapshot,
    *,
    allowed_tool_names: set[str] | None = None,
    forbidden_path_pattern: re.Pattern[str] | None = None,
    token_cost_estimate: int = 0,
) -> None:
    """Validate an action against safety, budget, and state preconditions."""
    _validate_safety(action, forbidden_path_pattern=forbidden_path_pattern)
    _validate_budget(action, snapshot, token_cost_estimate=token_cost_estimate)
    _validate_preconditions(action, snapshot, allowed_tool_names=allowed_tool_names)


def _validate_safety(
    action: Action,
    *,
    forbidden_path_pattern: re.Pattern[str] | None = None,
) -> None:
    if isinstance(action, ReadAction):
        if forbidden_path_pattern and forbidden_path_pattern.search(action.source_id):
            raise SafetyError(
                f"Read target matches forbidden path pattern: {action.source_id}"
            )
    if isinstance(action, SearchAction):
        if not action.query or not action.query.strip():
            raise SafetyError("Search query must be non-empty")


def _validate_budget(
    action: Action, snapshot: HarnessSnapshot, *, token_cost_estimate: int
) -> None:
    if snapshot.budget is None:
        return
    budget = snapshot.budget
    if budget.tokens_available < token_cost_estimate:
        raise BudgetError(
            f"Token budget exceeded: need {token_cost_estimate}, "
            f"have {budget.tokens_available}"
        )
    if isinstance(action, SearchAction | ReadAction):
        if budget.max_tool_calls is not None and budget.tool_calls_used >= budget.max_tool_calls:
            raise BudgetError(
                f"Tool call budget exceeded: {budget.tool_calls_used}/{budget.max_tool_calls}"
            )


def _validate_preconditions(
    action: Action,
    snapshot: HarnessSnapshot,
    *,
    allowed_tool_names: set[str] | None = None,
) -> None:
    if isinstance(action, DecomposeAction):
        existing_tasks = {t.task_id for t in snapshot.tasks}
        if action.task_id in existing_tasks:
            raise PreconditionError(f"Task already exists: {action.task_id}")
        if action.parent_task_id is not None and action.parent_task_id not in existing_tasks:
            raise PreconditionError(
                f"Parent task does not exist: {action.parent_task_id}"
            )
        missing_deps = set(action.dependency_ids) - existing_tasks
        if missing_deps:
            raise PreconditionError(f"Missing dependencies: {sorted(missing_deps)}")

    elif isinstance(action, CurateAction):
        existing_candidates = {c.item_id for c in snapshot.candidates}
        missing = set(action.candidate_ids) - existing_candidates
        if missing:
            raise PreconditionError(
                f"Curate references unknown candidates: {sorted(missing)}"
            )
        existing_curated = {c.item_id for c in snapshot.curated_items}
        if action.curated_item_id in existing_curated:
            raise PreconditionError(
                f"Curated item already exists: {action.curated_item_id}"
            )

    elif isinstance(action, DiscardAction):
        existing_candidates = {c.item_id for c in snapshot.candidates}
        missing = set(action.candidate_ids) - existing_candidates
        if missing:
            raise PreconditionError(
                f"Discard references unknown candidates: {sorted(missing)}"
            )

    elif isinstance(action, VerifyAction):
        existing_links = {link.link_id for link in snapshot.evidence_links}
        if action.link_id not in existing_links:
            raise PreconditionError(f"Evidence link does not exist: {action.link_id}")

    elif isinstance(action, ReviseClaimAction):
        existing_claims = {c.claim_id for c in snapshot.claims}
        if action.claim_id not in existing_claims:
            raise PreconditionError(f"Claim does not exist: {action.claim_id}")
        if action.new_content is None and action.new_status is None:
            raise PreconditionError("ReviseClaim must change content or status")

    elif isinstance(action, PruneSemanticAction):
        existing_curated = {c.item_id for c in snapshot.curated_items}
        unknown = set(action.retained_item_ids) - existing_curated
        if unknown:
            raise PreconditionError(
                f"Prune references unknown curated items: {sorted(unknown)}"
            )

    elif isinstance(action, SubmitAction):
        if not action.content or not action.content.strip():
            raise PreconditionError("Submission content must be non-empty")

    elif isinstance(action, SearchAction | ReadAction):
        if allowed_tool_names is not None:
            tool_name = "web_search" if isinstance(action, SearchAction) else "read_file"
            if tool_name not in allowed_tool_names:
                raise PreconditionError(f"Tool not allowed: {tool_name}")


# ---------------------------------------------------------------------------
# Action -> event planning
# ---------------------------------------------------------------------------


def plan_events(
    action: Action,
    snapshot: HarnessSnapshot,
    *,
    action_id: str,
    actor: str = "policy",
) -> list[HarnessEvent]:
    """Convert a validated action into one or more harness events."""
    sequence = snapshot.sequence + 1
    events: list[HarnessEvent] = []

    if isinstance(action, DecomposeAction):
        events.append(
            HarnessEvent.create(
                event_id=EventId(f"evt-{action_id}"),
                trajectory_id=snapshot.trajectory_id,
                sequence=sequence,
                event_type="task_decomposed",
                payload=TaskDecomposedPayload(
                    parent_task_id=TaskId(action.parent_task_id)
                    if action.parent_task_id is not None
                    else None,
                    task_id=TaskId(action.task_id),
                    description=action.description,
                    dependency_ids=tuple(TaskId(t) for t in action.dependency_ids),
                ),
                idempotency_key=f"{action_id}-decompose",
                actor=actor,
                action_id=action_id,
            )
        )

    elif isinstance(action, SearchAction):
        events.append(
            HarnessEvent.create(
                event_id=EventId(f"evt-{action_id}"),
                trajectory_id=snapshot.trajectory_id,
                sequence=sequence,
                event_type="tool_invoked",
                payload=ToolInvokedPayload(
                    invocation_id=ToolInvocationId(f"tool-{action_id}"),
                    parent_task_id=None,
                    tool_name="web_search",
                    arguments=(("query", action.query),),
                    dependency_ids=(),
                    deadline=None,
                ),
                idempotency_key=f"{action_id}-search",
                actor=actor,
                action_id=action_id,
            )
        )

    elif isinstance(action, ReadAction):
        events.append(
            HarnessEvent.create(
                event_id=EventId(f"evt-{action_id}"),
                trajectory_id=snapshot.trajectory_id,
                sequence=sequence,
                event_type="tool_invoked",
                payload=ToolInvokedPayload(
                    invocation_id=ToolInvocationId(f"tool-{action_id}"),
                    parent_task_id=None,
                    tool_name="read_file",
                    arguments=(("source_id", action.source_id),),
                    dependency_ids=(),
                    deadline=None,
                ),
                idempotency_key=f"{action_id}-read",
                actor=actor,
                action_id=action_id,
            )
        )

    elif isinstance(action, CurateAction):
        events.append(
            HarnessEvent.create(
                event_id=EventId(f"evt-{action_id}"),
                trajectory_id=snapshot.trajectory_id,
                sequence=sequence,
                event_type="evidence_curated",
                payload=EvidenceCuratedPayload(
                    candidate_ids=tuple(ItemId(c) for c in action.candidate_ids),
                    curated_item_id=ItemId(action.curated_item_id),
                    content=action.content,
                    priority=action.priority,
                    source_ids=action.source_ids,
                ),
                idempotency_key=f"{action_id}-curate",
                actor=actor,
                action_id=action_id,
            )
        )

    elif isinstance(action, DiscardAction):
        events.append(
            HarnessEvent.create(
                event_id=EventId(f"evt-{action_id}"),
                trajectory_id=snapshot.trajectory_id,
                sequence=sequence,
                event_type="action_committed",
                payload=ActionCommittedPayload(
                    action_type="discard",
                    action_params=(
                        ("candidate_ids", ",".join(action.candidate_ids)),
                        ("reason", action.reason),
                    ),
                ),
                idempotency_key=f"{action_id}-discard",
                actor=actor,
                action_id=action_id,
            )
        )

    elif isinstance(action, VerifyAction):
        events.append(
            HarnessEvent.create(
                event_id=EventId(f"evt-{action_id}"),
                trajectory_id=snapshot.trajectory_id,
                sequence=sequence,
                event_type="evidence_verified",
                payload=EvidenceVerifiedPayload(
                    link_id=LinkId(action.link_id),
                    verification_id=VerificationId(action.verification_id),
                    result=action.result,
                    details=action.details,
                ),
                idempotency_key=f"{action_id}-verify",
                actor=actor,
                action_id=action_id,
            )
        )

    elif isinstance(action, ReviseClaimAction):
        events.append(
            HarnessEvent.create(
                event_id=EventId(f"evt-{action_id}"),
                trajectory_id=snapshot.trajectory_id,
                sequence=sequence,
                event_type="claim_revised",
                payload=ClaimRevisedPayload(
                    claim_id=ClaimId(action.claim_id),
                    new_content=action.new_content,
                    new_status=action.new_status,
                ),
                idempotency_key=f"{action_id}-revise",
                actor=actor,
                action_id=action_id,
            )
        )

    elif isinstance(action, PruneSemanticAction):
        events.append(
            HarnessEvent.create(
                event_id=EventId(f"evt-{action_id}"),
                trajectory_id=snapshot.trajectory_id,
                sequence=sequence,
                event_type="context_pruned",
                payload=ContextPrunedPayload(
                    retained_item_ids=tuple(ItemId(i) for i in action.retained_item_ids),
                    new_epoch_id=action.new_epoch_id,
                ),
                idempotency_key=f"{action_id}-prune",
                actor=actor,
                action_id=action_id,
            )
        )

    elif isinstance(action, RequestContextAction):
        events.append(
            HarnessEvent.create(
                event_id=EventId(f"evt-{action_id}"),
                trajectory_id=snapshot.trajectory_id,
                sequence=sequence,
                event_type="action_committed",
                payload=ActionCommittedPayload(
                    action_type="request_context",
                    action_params=(
                        ("context_type", action.context_type),
                        *action.parameters,
                    ),
                ),
                idempotency_key=f"{action_id}-request-context",
                actor=actor,
                action_id=action_id,
            )
        )

    elif isinstance(action, SubmitAction):
        events.append(
            HarnessEvent.create(
                event_id=EventId(f"evt-{action_id}"),
                trajectory_id=snapshot.trajectory_id,
                sequence=sequence,
                event_type="submission_recorded",
                payload=SubmissionRecordedPayload(
                    submission_id=SubmissionId(action.submission_id),
                    content=action.content,
                    source_ids=action.source_ids,
                ),
                idempotency_key=f"{action_id}-submit",
                actor=actor,
                action_id=action_id,
            )
        )

    else:
        raise ActionError(f"Unknown action type: {type(action).__name__}")

    return events
