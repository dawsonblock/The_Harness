"""Immutable domain schemas for the harness control plane."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from rfsn_agent.cas import ContentAddressedStore
from rfsn_agent.common import (
    canonical_json,
    dataclass_from_dict,
    dataclass_to_dict,
    hash_content,
    utc_now,
)
from rfsn_agent.types import (
    ClaimId,
    ClaimStatus,
    ContentHash,
    ItemId,
    LinkId,
    SubmissionId,
    TaskId,
    TaskStatus,
    ToolInvocationId,
    ToolStatus,
    TrajectoryId,
    VerificationId,
    VerificationResult,
    VerificationStatus,
)


@dataclass(frozen=True, slots=True)
class Provenance:
    """Immutable source-of-origin metadata attached to every domain object."""

    actor: str
    action_id: str
    event_id: str | None = None
    parent_ids: tuple[str, ...] = field(default_factory=tuple)
    metadata: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    def with_event(self, event_id: str) -> Provenance:
        """Return a new provenance record stamped with an event id."""
        return Provenance(
            actor=self.actor,
            action_id=self.action_id,
            event_id=event_id,
            parent_ids=self.parent_ids,
            metadata=self.metadata,
        )


_DEFAULT_PROVENANCE = Provenance(actor="system", action_id="init")


@dataclass(frozen=True, slots=True)
class ContentReference:
    """A pointer to content stored in an external content-addressed store."""

    content_hash: ContentHash
    byte_length: int


@dataclass(frozen=True, slots=True)
class CandidateItem:
    """A raw item retrieved from an external source, before curation."""

    item_id: ItemId
    trajectory_id: TrajectoryId
    source_id: str
    retrieval_query: str
    content: str
    content_hash: ContentHash
    metadata: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    created_at: datetime = field(default_factory=utc_now)
    provenance: Provenance = field(default_factory=lambda: _DEFAULT_PROVENANCE)
    content_ref: ContentReference | None = None

    def __post_init__(self) -> None:
        if self.content_ref is None:
            expected = hash_content(self.content)
            if expected != self.content_hash:
                raise ValueError(
                    f"CandidateItem {self.item_id}: content_hash mismatch: "
                    f"expected {expected}, got {self.content_hash}"
                )

    def resolve_content(self, store: ContentAddressedStore) -> str:
        if self.content_ref is not None:
            return store.get_text(self.content_ref.content_hash)
        return self.content

    @classmethod
    def create(
        cls,
        *,
        item_id: ItemId,
        trajectory_id: TrajectoryId,
        source_id: str,
        retrieval_query: str,
        content: str,
        metadata: tuple[tuple[str, str], ...] | None = None,
        provenance: Provenance | None = None,
    ) -> CandidateItem:
        return cls(
            item_id=item_id,
            trajectory_id=trajectory_id,
            source_id=source_id,
            retrieval_query=retrieval_query,
            content=content,
            content_hash=hash_content(content),
            metadata=metadata or (),
            provenance=provenance or _DEFAULT_PROVENANCE,
        )

    @classmethod
    def create_with_cas(
        cls,
        *,
        item_id: ItemId,
        trajectory_id: TrajectoryId,
        source_id: str,
        retrieval_query: str,
        content: str,
        metadata: tuple[tuple[str, str], ...] | None = None,
        provenance: Provenance | None = None,
        cas: ContentAddressedStore,
    ) -> CandidateItem:
        content_hash = hash_content(content)
        cas.put(content)
        return cls(
            item_id=item_id,
            trajectory_id=trajectory_id,
            source_id=source_id,
            retrieval_query=retrieval_query,
            content="",
            content_hash=content_hash,
            metadata=metadata or (),
            provenance=provenance or _DEFAULT_PROVENANCE,
            content_ref=ContentReference(
                content_hash=content_hash,
                byte_length=len(content.encode("utf-8")),
            ),
        )


@dataclass(frozen=True, slots=True)
class CuratedItem:
    """Evidence selected from candidates and promoted into working memory."""

    item_id: ItemId
    trajectory_id: TrajectoryId
    candidate_ids: tuple[ItemId, ...]
    content: str
    content_hash: ContentHash
    priority: int = 0
    source_ids: tuple[str, ...] = field(default_factory=tuple)
    created_at: datetime = field(default_factory=utc_now)
    provenance: Provenance = field(default_factory=lambda: _DEFAULT_PROVENANCE)

    def __post_init__(self) -> None:
        expected = hash_content(self.content)
        if expected != self.content_hash:
            raise ValueError(
                f"CuratedItem {self.item_id}: content_hash mismatch: "
                f"expected {expected}, got {self.content_hash}"
            )

    @classmethod
    def create(
        cls,
        *,
        item_id: ItemId,
        trajectory_id: TrajectoryId,
        candidate_ids: tuple[ItemId, ...],
        content: str,
        priority: int = 0,
        source_ids: tuple[str, ...] | None = None,
        provenance: Provenance | None = None,
    ) -> CuratedItem:
        return cls(
            item_id=item_id,
            trajectory_id=trajectory_id,
            candidate_ids=candidate_ids,
            content=content,
            content_hash=hash_content(content),
            priority=priority,
            source_ids=source_ids or (),
            provenance=provenance or _DEFAULT_PROVENANCE,
        )


@dataclass(frozen=True, slots=True)
class Claim:
    """A proposition the agent has stated and may verify."""

    claim_id: ClaimId
    trajectory_id: TrajectoryId
    content: str
    content_hash: ContentHash
    status: ClaimStatus = ClaimStatus.STATED
    evidence_link_ids: tuple[LinkId, ...] = field(default_factory=tuple)
    created_at: datetime = field(default_factory=utc_now)
    provenance: Provenance = field(default_factory=lambda: _DEFAULT_PROVENANCE)

    def __post_init__(self) -> None:
        expected = hash_content(self.content)
        if expected != self.content_hash:
            raise ValueError(
                f"Claim {self.claim_id}: content_hash mismatch: "
                f"expected {expected}, got {self.content_hash}"
            )

    @classmethod
    def create(
        cls,
        *,
        claim_id: ClaimId,
        trajectory_id: TrajectoryId,
        content: str,
        status: ClaimStatus = ClaimStatus.STATED,
        evidence_link_ids: tuple[LinkId, ...] | None = None,
        provenance: Provenance | None = None,
    ) -> Claim:
        return cls(
            claim_id=claim_id,
            trajectory_id=trajectory_id,
            content=content,
            content_hash=hash_content(content),
            status=status,
            evidence_link_ids=evidence_link_ids or (),
            provenance=provenance or _DEFAULT_PROVENANCE,
        )

    def with_status(self, status: ClaimStatus, provenance: Provenance) -> Claim:
        """Return a new claim with an updated status."""
        return Claim(
            claim_id=self.claim_id,
            trajectory_id=self.trajectory_id,
            content=self.content,
            content_hash=self.content_hash,
            status=status,
            evidence_link_ids=self.evidence_link_ids,
            created_at=self.created_at,
            provenance=provenance,
        )


@dataclass(frozen=True, slots=True)
class EvidenceLink:
    """A directional link between a claim and a curated evidence item."""

    link_id: LinkId
    trajectory_id: TrajectoryId
    claim_id: ClaimId
    curated_item_id: ItemId
    relationship: str  # supports, contradicts, neutral
    strength: float
    current_status: VerificationStatus = VerificationStatus.UNVERIFIED
    verification_id: VerificationId | None = None
    created_at: datetime = field(default_factory=utc_now)
    provenance: Provenance = field(default_factory=lambda: _DEFAULT_PROVENANCE)

    def __post_init__(self) -> None:
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(
                f"EvidenceLink {self.link_id}: strength must be in [0, 1], got {self.strength}"
            )


@dataclass(frozen=True, slots=True)
class VerificationRecord:
    """A record of an attempt to verify an evidence link."""

    record_id: VerificationId
    trajectory_id: TrajectoryId
    link_id: LinkId
    claim_id: ClaimId
    method: str
    result: VerificationResult
    details: str
    details_hash: ContentHash
    created_at: datetime = field(default_factory=utc_now)
    provenance: Provenance = field(default_factory=lambda: _DEFAULT_PROVENANCE)

    def __post_init__(self) -> None:
        expected = hash_content(self.details)
        if expected != self.details_hash:
            raise ValueError(
                f"VerificationRecord {self.record_id}: details_hash mismatch: "
                f"expected {expected}, got {self.details_hash}"
            )

    @classmethod
    def create(
        cls,
        *,
        record_id: VerificationId,
        trajectory_id: TrajectoryId,
        link_id: LinkId,
        claim_id: ClaimId,
        method: str,
        result: VerificationResult,
        details: str,
        provenance: Provenance | None = None,
    ) -> VerificationRecord:
        return cls(
            record_id=record_id,
            trajectory_id=trajectory_id,
            link_id=link_id,
            claim_id=claim_id,
            method=method,
            result=result,
            details=details,
            details_hash=hash_content(details),
            provenance=provenance or _DEFAULT_PROVENANCE,
        )


@dataclass(frozen=True, slots=True)
class TaskNode:
    """A unit of work in the agent's decomposition DAG."""

    task_id: TaskId
    trajectory_id: TrajectoryId
    parent_id: TaskId | None
    description: str
    status: TaskStatus = TaskStatus.PENDING
    dependency_ids: tuple[TaskId, ...] = field(default_factory=tuple)
    created_at: datetime = field(default_factory=utc_now)
    completed_at: datetime | None = None
    provenance: Provenance = field(default_factory=lambda: _DEFAULT_PROVENANCE)


@dataclass(frozen=True, slots=True)
class BudgetLedger:
    """Token, tool-call, and wall-time budget accounting."""

    trajectory_id: TrajectoryId
    max_tokens: int
    tokens_used: int = 0
    tokens_reserved: int = 0
    max_tool_calls: int | None = None
    tool_calls_used: int = 0
    max_wall_seconds: float | None = None
    wall_seconds_used: float = 0.0

    @property
    def tokens_available(self) -> int:
        return max(0, self.max_tokens - self.tokens_used - self.tokens_reserved)

    def reserve(self, tokens: int) -> BudgetLedger:
        if tokens < 0:
            raise ValueError("Cannot reserve negative tokens")
        return BudgetLedger(
            trajectory_id=self.trajectory_id,
            max_tokens=self.max_tokens,
            tokens_used=self.tokens_used,
            tokens_reserved=self.tokens_reserved + tokens,
            max_tool_calls=self.max_tool_calls,
            tool_calls_used=self.tool_calls_used,
            max_wall_seconds=self.max_wall_seconds,
            wall_seconds_used=self.wall_seconds_used,
        )

    def spend(
        self, tokens: int, tool_calls: int = 0, wall_seconds: float = 0.0
    ) -> BudgetLedger:
        if tokens < 0 or tool_calls < 0 or wall_seconds < 0:
            raise ValueError("Cannot spend negative resources")
        return BudgetLedger(
            trajectory_id=self.trajectory_id,
            max_tokens=self.max_tokens,
            tokens_used=self.tokens_used + tokens,
            tokens_reserved=max(0, self.tokens_reserved - tokens),
            max_tool_calls=self.max_tool_calls,
            tool_calls_used=self.tool_calls_used + tool_calls,
            max_wall_seconds=self.max_wall_seconds,
            wall_seconds_used=self.wall_seconds_used + wall_seconds,
        )


@dataclass(frozen=True, slots=True)
class ToolInvocation:
    """A scheduled tool call with dependency and deadline metadata."""

    invocation_id: ToolInvocationId
    trajectory_id: TrajectoryId
    action_id: str
    parent_task_id: TaskId | None
    tool_name: str
    arguments: tuple[tuple[str, str], ...]
    arguments_hash: ContentHash
    dependency_ids: tuple[ToolInvocationId, ...] = field(default_factory=tuple)
    deadline: datetime | None = None
    created_at: datetime = field(default_factory=utc_now)
    provenance: Provenance = field(default_factory=lambda: _DEFAULT_PROVENANCE)

    def __post_init__(self) -> None:
        expected = hash_content(canonical_json(dict(self.arguments)))
        if expected != self.arguments_hash:
            raise ValueError(
                f"ToolInvocation {self.invocation_id}: arguments_hash mismatch: "
                f"expected {expected}, got {self.arguments_hash}"
            )

    @classmethod
    def create(
        cls,
        *,
        invocation_id: ToolInvocationId,
        trajectory_id: TrajectoryId,
        action_id: str,
        parent_task_id: TaskId | None,
        tool_name: str,
        arguments: tuple[tuple[str, str], ...],
        dependency_ids: tuple[ToolInvocationId, ...] | None = None,
        deadline: datetime | None = None,
        provenance: Provenance | None = None,
    ) -> ToolInvocation:
        return cls(
            invocation_id=invocation_id,
            trajectory_id=trajectory_id,
            action_id=action_id,
            parent_task_id=parent_task_id,
            tool_name=tool_name,
            arguments=arguments,
            arguments_hash=hash_content(canonical_json(dict(arguments))),
            dependency_ids=dependency_ids or (),
            deadline=deadline,
            provenance=provenance or _DEFAULT_PROVENANCE,
        )


@dataclass(frozen=True, slots=True)
class ToolResult:
    """The result of a tool invocation."""

    result_id: ToolInvocationId
    invocation_id: ToolInvocationId
    trajectory_id: TrajectoryId
    status: ToolStatus
    content: str
    content_hash: ContentHash
    received_at: datetime = field(default_factory=utc_now)
    provenance: Provenance = field(default_factory=lambda: _DEFAULT_PROVENANCE)
    content_ref: ContentReference | None = None

    def __post_init__(self) -> None:
        if self.content_ref is None:
            expected = hash_content(self.content)
            if expected != self.content_hash:
                raise ValueError(
                    f"ToolResult {self.result_id}: content_hash mismatch: "
                    f"expected {expected}, got {self.content_hash}"
                )

    def resolve_content(self, store: ContentAddressedStore) -> str:
        if self.content_ref is not None:
            return store.get_text(self.content_ref.content_hash)
        return self.content

    @classmethod
    def create(
        cls,
        *,
        result_id: ToolInvocationId,
        invocation_id: ToolInvocationId,
        trajectory_id: TrajectoryId,
        status: ToolStatus,
        content: str,
        provenance: Provenance | None = None,
    ) -> ToolResult:
        return cls(
            result_id=result_id,
            invocation_id=invocation_id,
            trajectory_id=trajectory_id,
            status=status,
            content=content,
            content_hash=hash_content(content),
            provenance=provenance or _DEFAULT_PROVENANCE,
        )

    @classmethod
    def create_with_cas(
        cls,
        *,
        result_id: ToolInvocationId,
        invocation_id: ToolInvocationId,
        trajectory_id: TrajectoryId,
        status: ToolStatus,
        content: str,
        provenance: Provenance | None = None,
        cas: ContentAddressedStore,
    ) -> ToolResult:
        content_hash = hash_content(content)
        cas.put(content)
        return cls(
            result_id=result_id,
            invocation_id=invocation_id,
            trajectory_id=trajectory_id,
            status=status,
            content="",
            content_hash=content_hash,
            provenance=provenance or _DEFAULT_PROVENANCE,
            content_ref=ContentReference(
                content_hash=content_hash,
                byte_length=len(content.encode("utf-8")),
            ),
        )


@dataclass(frozen=True, slots=True)
class SubmissionRecord:
    """A final or intermediate submission produced by the agent."""

    submission_id: SubmissionId
    trajectory_id: TrajectoryId
    content: str
    content_hash: ContentHash
    source_ids: tuple[str, ...] = field(default_factory=tuple)
    submitted_at: datetime = field(default_factory=utc_now)
    provenance: Provenance = field(default_factory=lambda: _DEFAULT_PROVENANCE)
    content_ref: ContentReference | None = None

    def __post_init__(self) -> None:
        if self.content_ref is None:
            expected = hash_content(self.content)
            if expected != self.content_hash:
                raise ValueError(
                    f"SubmissionRecord {self.submission_id}: content_hash mismatch: "
                    f"expected {expected}, got {self.content_hash}"
                )

    def resolve_content(self, store: ContentAddressedStore) -> str:
        if self.content_ref is not None:
            return store.get_text(self.content_ref.content_hash)
        return self.content

    @classmethod
    def create(
        cls,
        *,
        submission_id: SubmissionId,
        trajectory_id: TrajectoryId,
        content: str,
        source_ids: tuple[str, ...] | None = None,
        provenance: Provenance | None = None,
    ) -> SubmissionRecord:
        return cls(
            submission_id=submission_id,
            trajectory_id=trajectory_id,
            content=content,
            content_hash=hash_content(content),
            source_ids=source_ids or (),
            provenance=provenance or _DEFAULT_PROVENANCE,
        )

    @classmethod
    def create_with_cas(
        cls,
        *,
        submission_id: SubmissionId,
        trajectory_id: TrajectoryId,
        content: str,
        source_ids: tuple[str, ...] | None = None,
        provenance: Provenance | None = None,
        cas: ContentAddressedStore,
    ) -> SubmissionRecord:
        content_hash = hash_content(content)
        cas.put(content)
        return cls(
            submission_id=submission_id,
            trajectory_id=trajectory_id,
            content="",
            content_hash=content_hash,
            source_ids=source_ids or (),
            provenance=provenance or _DEFAULT_PROVENANCE,
            content_ref=ContentReference(
                content_hash=content_hash,
                byte_length=len(content.encode("utf-8")),
            ),
        )


@dataclass(frozen=True, slots=True)
class HarnessSnapshot:
    """A derived, immutable view of harness state at a specific sequence."""

    trajectory_id: TrajectoryId
    epoch_id: str
    sequence: int
    state_hash: ContentHash
    last_event_hash: ContentHash | None
    last_signature: ContentHash | None = None
    created_at: datetime = field(default_factory=utc_now)
    candidates: tuple[CandidateItem, ...] = field(default_factory=tuple)
    curated_items: tuple[CuratedItem, ...] = field(default_factory=tuple)
    claims: tuple[Claim, ...] = field(default_factory=tuple)
    evidence_links: tuple[EvidenceLink, ...] = field(default_factory=tuple)
    verification_records: tuple[VerificationRecord, ...] = field(default_factory=tuple)
    tasks: tuple[TaskNode, ...] = field(default_factory=tuple)
    budget: BudgetLedger | None = None
    submissions: tuple[SubmissionRecord, ...] = field(default_factory=tuple)
    tool_invocations: tuple[ToolInvocation, ...] = field(default_factory=tuple)
    tool_results: tuple[ToolResult, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        expected = self.compute_state_hash()
        if expected != self.state_hash:
            raise ValueError(
                f"HarnessSnapshot: state_hash mismatch: expected {expected}, got {self.state_hash}"
            )

    def compute_state_hash(self) -> str:
        """Compute a deterministic hash over all deterministic fields."""
        payload = {
            "trajectory_id": self.trajectory_id,
            "epoch_id": self.epoch_id,
            "sequence": self.sequence,
            "last_event_hash": self.last_event_hash,
            "last_signature": self.last_signature,
            "candidates": [canonical_json(c) for c in self.candidates],
            "curated_items": [canonical_json(c) for c in self.curated_items],
            "claims": [canonical_json(c) for c in self.claims],
            "evidence_links": [canonical_json(c) for c in self.evidence_links],
            "verification_records": [canonical_json(c) for c in self.verification_records],
            "tasks": [canonical_json(c) for c in self.tasks],
            "budget": canonical_json(self.budget) if self.budget else None,
            "submissions": [canonical_json(c) for c in self.submissions],
            "tool_invocations": [canonical_json(c) for c in self.tool_invocations],
            "tool_results": [canonical_json(c) for c in self.tool_results],
        }
        return hash_content(canonical_json(payload))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the snapshot to a dictionary."""
        return dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HarnessSnapshot:
        """Deserialize a snapshot from a dictionary and validate its hash."""
        return dataclass_from_dict(cls, data)

    @classmethod
    def create(
        cls,
        *,
        trajectory_id: TrajectoryId,
        epoch_id: str,
        sequence: int,
        last_event_hash: ContentHash | None = None,
        **kwargs: Any,
    ) -> HarnessSnapshot:
        """Create a snapshot with a correctly computed state hash."""
        fields = {
            "trajectory_id": trajectory_id,
            "epoch_id": epoch_id,
            "sequence": sequence,
            "last_event_hash": last_event_hash,
            "last_signature": kwargs.get("last_signature", None),
            "state_hash": "",
            "created_at": kwargs.get("created_at", utc_now()),
            "candidates": kwargs.get("candidates", ()),
            "curated_items": kwargs.get("curated_items", ()),
            "claims": kwargs.get("claims", ()),
            "evidence_links": kwargs.get("evidence_links", ()),
            "verification_records": kwargs.get("verification_records", ()),
            "tasks": kwargs.get("tasks", ()),
            "budget": kwargs.get("budget", None),
            "submissions": kwargs.get("submissions", ()),
            "tool_invocations": kwargs.get("tool_invocations", ()),
            "tool_results": kwargs.get("tool_results", ()),
        }
        transient = object.__new__(cls)
        for name, value in fields.items():
            object.__setattr__(transient, name, value)
        fields["state_hash"] = transient.compute_state_hash()
        return cls(**fields)
