"""Deterministic context compiler: harness state -> model-visible ContextPacket."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Protocol

from rfsn_agent.common import canonical_json, hash_content
from rfsn_agent.domain import (
    CandidateItem,
    CuratedItem,
    HarnessSnapshot,
    TaskNode,
    ToolResult,
)
from rfsn_agent.types import ClaimStatus


class TokenCounter(Protocol):
    """Protocol for deterministic token counting."""

    def count(self, text: str) -> int:
        ...


class WhitespaceTokenCounter:
    """Simple token counter for testing and baseline work.

    Production compilers should inject a model-specific tokenizer.
    """

    def count(self, text: str) -> int:
        return len(text.split())


class TiktokenTokenCounter:
    """Token counter backed by tiktoken, falling back to whitespace counting."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self.encoding_name = encoding_name
        try:
            import tiktoken  # type: ignore[import-not-found]

            self._encoding = tiktoken.get_encoding(encoding_name)
        except Exception:
            self._encoding = None

    def count(self, text: str) -> int:
        if self._encoding is None:
            return len(text.split())
        return len(self._encoding.encode(text))


class HuggingFaceTokenizerCounter:
    """Token counter backed by an injected HuggingFace tokenizer.

    The tokenizer may be a HuggingFace ``PreTrainedTokenizerBase`` or any object
    with a compatible ``__call__(text, add_special_tokens=False)`` interface.
    If the tokenizer is unavailable or raises, counting falls back to whitespace.
    """

    def __init__(
        self,
        tokenizer: Callable[[str], Any] | None = None,
        model_name: str | None = None,
    ) -> None:
        self.model_name = model_name
        if tokenizer is not None:
            self._tokenizer = tokenizer
            return
        if model_name is None:
            self._tokenizer = None
            return
        try:
            from transformers import AutoTokenizer  # type: ignore[import-not-found]

            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception:
            self._tokenizer = None

    def count(self, text: str) -> int:
        if self._tokenizer is None:
            return len(text.split())
        try:
            return len(self._tokenizer(text, add_special_tokens=False)["input_ids"])
        except Exception:
            return len(text.split())


class TrustedRole(Enum):
    """Trust boundary roles for context segments."""

    SYSTEM = auto()
    USER = auto()
    ASSISTANT = auto()
    TOOL = auto()


class ContextSelector(Protocol):
    """Protocol for selecting context segments from a snapshot."""

    def select(
        self,
        snapshot: HarnessSnapshot,
        config: CompilerConfig,
        token_counter: TokenCounter,
    ) -> list[ContextSegment]:
        ...


class ContextRenderer(Protocol):
    """Protocol for rendering context segments into a model-specific string."""

    def render(self, segments: list[ContextSegment], model_type: str = "gpt4") -> str:
        ...


class DefaultContextSelector:
    """Default selector that replicates the original compile_context selection logic."""

    def select(
        self,
        snapshot: HarnessSnapshot,
        config: CompilerConfig,
        token_counter: TokenCounter,
    ) -> list[ContextSegment]:
        segments = _collect_segments(snapshot, config)

        # Exact deduplication by content hash (keep first occurrence).
        seen_hashes: set[str] = set()
        deduped: list[ContextSegment] = []
        for segment in segments:
            if segment.content_hash in seen_hashes:
                continue
            seen_hashes.add(segment.content_hash)
            deduped.append(segment)

        # Allocate tokens and truncate.
        allocated = _allocate_tokens(deduped, config, token_counter)

        # Add retrieval handles for omitted material.
        final_segments = _add_retrieval_handles(
            allocated, snapshot, config, token_counter
        )
        return final_segments


class JinjaContextRenderer:
    """Renderer that formats segments for specific LLM formats."""

    def render(self, segments: list[ContextSegment], model_type: str = "gpt4") -> str:
        if model_type == "llama3":
            parts = ["<|begin_of_text|>"]
            for seg in segments:
                parts.append(f'<segment kind="{seg.kind}">')
                parts.append(seg.content)
                parts.append("</segment>")
            parts.append("<|end_of_text|>")
            return "\n".join(parts)
        elif model_type == "gpt4":
            parts = []
            for seg in segments:
                parts.append(f"### {seg.kind}\n{seg.content}")
            return "\n\n".join(parts)
        elif model_type == "claude":
            parts = []
            for seg in segments:
                parts.append(f"[{seg.kind}]\n{seg.content}")
            return "\n\n".join(parts)
        else:
            parts = []
            for seg in segments:
                parts.append(f"### {seg.kind}\n{seg.content}")
            return "\n\n".join(parts)


@dataclass(frozen=True, slots=True)
class CompilerConfig:
    """Budget and priority configuration for context compilation."""

    max_total_tokens: int
    task_constraints_budget: int | None = None
    plan_budget: int | None = None
    verified_evidence_budget: int | None = None
    active_evidence_budget: int | None = None
    contradiction_notes_budget: int | None = None
    tool_results_budget: int | None = None
    candidate_preview_budget: int | None = None
    retrieval_handles_budget: int | None = None

    # Section priorities: higher values are evicted last.
    task_constraints_priority: int = 100
    plan_priority: int = 95
    verified_evidence_priority: int = 90
    contradiction_notes_priority: int = 85
    active_evidence_priority: int = 80
    tool_results_priority: int = 70
    retrieval_handles_priority: int = 60
    candidate_preview_priority: int = 40

    include_candidate_previews: bool = False
    max_tool_results: int = 5
    max_candidate_previews: int = 3


@dataclass(frozen=True, slots=True)
class ContextSegment:
    """A single deterministic unit inside a ContextPacket."""

    segment_id: str
    kind: str
    priority: int
    content: str
    token_count: int
    source_ids: tuple[str, ...]
    content_hash: str
    role: TrustedRole = TrustedRole.SYSTEM

    def __post_init__(self) -> None:
        expected = hash_content(self.content)
        if expected != self.content_hash:
            raise ValueError(
                f"ContextSegment {self.segment_id}: content_hash mismatch: "
                f"expected {expected}, got {self.content_hash}"
            )

    @classmethod
    def create(
        cls,
        *,
        segment_id: str,
        kind: str,
        priority: int,
        content: str,
        source_ids: tuple[str, ...] | None = None,
        role: TrustedRole = TrustedRole.SYSTEM,
    ) -> ContextSegment:
        return cls(
            segment_id=segment_id,
            kind=kind,
            priority=priority,
            content=content,
            token_count=0,  # filled by compiler
            source_ids=source_ids or (),
            content_hash=hash_content(content),
            role=role,
        )

    def with_token_count(self, token_count: int) -> ContextSegment:
        return ContextSegment(
            segment_id=self.segment_id,
            kind=self.kind,
            priority=self.priority,
            content=self.content,
            token_count=token_count,
            source_ids=self.source_ids,
            content_hash=self.content_hash,
            role=self.role,
        )


@dataclass(frozen=True, slots=True)
class ContextPacket:
    """A compiled, deterministic package of model-visible context."""

    trajectory_id: str
    epoch_id: str
    state_sequence: int
    segments: tuple[ContextSegment, ...]
    total_tokens: int
    source_hashes: tuple[str, ...]
    packet_hash: str
    token_ids: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        expected = self._compute_packet_hash()
        if expected != self.packet_hash:
            raise ValueError(
                f"ContextPacket: packet_hash mismatch: expected {expected}, got {self.packet_hash}"
            )

    def _compute_packet_hash(self) -> str:
        payload = {
            "trajectory_id": self.trajectory_id,
            "epoch_id": self.epoch_id,
            "state_sequence": self.state_sequence,
            "segments": [canonical_json(s) for s in self.segments],
            "total_tokens": self.total_tokens,
            "source_hashes": sorted(self.source_hashes),
            "token_ids": list(self.token_ids) if self.token_ids is not None else None,
        }
        return hash_content(canonical_json(payload))

    @classmethod
    def create(
        cls,
        *,
        trajectory_id: str,
        epoch_id: str,
        state_sequence: int,
        segments: tuple[ContextSegment, ...],
        total_tokens: int,
        source_hashes: tuple[str, ...],
        token_ids: tuple[int, ...] | None = None,
    ) -> ContextPacket:
        # Use object.__new__ to compute hash before __post_init__ validation.
        fields: dict[str, Any] = {
            "trajectory_id": trajectory_id,
            "epoch_id": epoch_id,
            "state_sequence": state_sequence,
            "segments": segments,
            "total_tokens": total_tokens,
            "source_hashes": source_hashes,
            "packet_hash": "",
            "token_ids": token_ids,
        }
        transient = object.__new__(cls)
        for name, value in fields.items():
            object.__setattr__(transient, name, value)
        fields["packet_hash"] = transient._compute_packet_hash()
        return cls(**fields)


@dataclass(frozen=True, slots=True)
class ContextEpoch:
    """Describes prefix reuse between two context packets."""

    epoch_id: str
    parent_epoch_id: str | None
    packet_hash: str
    common_prefix_tokens: int
    invalidated_suffix_tokens: int
    cache_branch_id: str


# ---------------------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------------------


def compile_context(
    snapshot: HarnessSnapshot,
    config: CompilerConfig,
    token_counter: TokenCounter | None = None,
    previous_packet: ContextPacket | None = None,
    renderer: ContextRenderer | None = None,
    model_type: str = "gpt4",
) -> ContextPacket:
    """Compile a harness snapshot into a deterministic ContextPacket."""
    counter = token_counter or WhitespaceTokenCounter()

    # Stage 1-4: select segments using the default selector.
    selector = DefaultContextSelector()
    final_segments = selector.select(snapshot, config, counter)

    # Stage 5: render with roles if a renderer is provided.
    if renderer is not None:
        final_segments = _render_with_roles(
            final_segments, renderer, model_type, counter
        )

    # Stage 6: build packet.
    total_tokens = sum(s.token_count for s in final_segments)
    source_hashes = tuple(sorted({s.content_hash for s in final_segments}))
    packet = ContextPacket.create(
        trajectory_id=snapshot.trajectory_id,
        epoch_id=snapshot.epoch_id,
        state_sequence=snapshot.sequence,
        segments=tuple(final_segments),
        total_tokens=total_tokens,
        source_hashes=source_hashes,
    )

    # Stage 7: compute epoch reuse if a previous packet is supplied.
    if previous_packet is not None:
        _compute_epoch(previous_packet, packet)

    return packet


def _render_with_roles(
    segments: list[ContextSegment],
    renderer: ContextRenderer,
    model_type: str,
    counter: TokenCounter,
) -> list[ContextSegment]:
    """Group segments by role, insert boundary markers between role changes, and render."""
    if not segments:
        return segments

    marked_segments: list[ContextSegment] = []
    prev_role: TrustedRole | None = None
    for seg in segments:
        if prev_role is not None and seg.role != prev_role:
            boundary = _role_boundary_marker(seg.role)
            boundary_seg = ContextSegment.create(
                segment_id=f"role-boundary-{len(marked_segments)}",
                kind="role_boundary",
                priority=0,
                content=boundary,
                role=seg.role,
            )
            boundary_seg = boundary_seg.with_token_count(
                counter.count(boundary_seg.content)
            )
            marked_segments.append(boundary_seg)
        marked_segments.append(seg)
        prev_role = seg.role

    rendered = renderer.render(marked_segments, model_type)
    rendered_seg = ContextSegment.create(
        segment_id="rendered",
        kind="rendered",
        priority=0,
        content=rendered,
        role=TrustedRole.SYSTEM,
    )
    rendered_seg = rendered_seg.with_token_count(counter.count(rendered_seg.content))
    return [rendered_seg]


def _role_boundary_marker(role: TrustedRole) -> str:
    return f"<|{role.name.lower()}|>"


def _collect_segments(
    snapshot: HarnessSnapshot, config: CompilerConfig
) -> list[ContextSegment]:
    segments: list[ContextSegment] = []

    # Task constraints (reserved, never pruned).
    if snapshot.tasks or snapshot.budget is not None:
        segments.append(_task_constraints_segment(snapshot, config))

    # Current plan / unresolved tasks.
    unresolved = [
        t for t in snapshot.tasks if t.status.value in ("pending", "in_progress", "blocked")
    ]
    if unresolved:
        segments.append(_plan_segment(unresolved, config))

    # Verified evidence.
    verified_item_ids = _verified_curated_item_ids(snapshot)
    verified_items = [c for c in snapshot.curated_items if c.item_id in verified_item_ids]
    if verified_items:
        segments.append(_verified_evidence_segment(verified_items, config))

    # Active (unverified) evidence.
    unverified_items = [
        c for c in snapshot.curated_items if c.item_id not in verified_item_ids
    ]
    if unverified_items:
        segments.append(_active_evidence_segment(unverified_items, config))

    # Contradictions and withdrawn claims.
    segments.extend(_contradiction_segments(snapshot, config))

    # Tool results.
    recent_tool_results = snapshot.tool_results[-config.max_tool_results :]
    if recent_tool_results:
        segments.append(_tool_results_segment(recent_tool_results, config))

    # Candidate previews.
    if config.include_candidate_previews:
        recent_candidates = snapshot.candidates[-config.max_candidate_previews :]
        if recent_candidates:
            segments.append(_candidate_preview_segment(recent_candidates, config))

    return segments


def _task_constraints_segment(
    snapshot: HarnessSnapshot, config: CompilerConfig
) -> ContextSegment:
    parts: list[str] = []
    if snapshot.budget is not None:
        b = snapshot.budget
        parts.append(
            f"Budget: {b.tokens_available}/{b.max_tokens} tokens available, "
            f"{b.tool_calls_used} tool calls used."
        )
    for task in snapshot.tasks:
        parts.append(f"Task {task.task_id}: {task.description} ({task.status.value})")
    content = "\n".join(parts)
    return ContextSegment.create(
        segment_id="task-constraints",
        kind="task_constraints",
        priority=config.task_constraints_priority,
        content=content,
        role=TrustedRole.SYSTEM,
    )


def _plan_segment(unresolved: list[TaskNode], config: CompilerConfig) -> ContextSegment:
    lines = [f"- {t.task_id}: {t.description}" for t in unresolved]
    return ContextSegment.create(
        segment_id="plan",
        kind="plan",
        priority=config.plan_priority,
        content="Plan\n" + "\n".join(lines),
        role=TrustedRole.SYSTEM,
    )


def _verified_evidence_segment(
    items: list[CuratedItem], config: CompilerConfig
) -> ContextSegment:
    lines = [f"- {item.item_id}: {item.content}" for item in items]
    return ContextSegment.create(
        segment_id="verified-evidence",
        kind="verified_evidence",
        priority=config.verified_evidence_priority,
        content="Verified evidence\n" + "\n".join(lines),
        source_ids=tuple(i.item_id for i in items),
        role=TrustedRole.SYSTEM,
    )


def _active_evidence_segment(
    items: list[CuratedItem], config: CompilerConfig
) -> ContextSegment:
    lines = [f"- {item.item_id}: {item.content}" for item in items]
    return ContextSegment.create(
        segment_id="active-evidence",
        kind="active_evidence",
        priority=config.active_evidence_priority,
        content="Active evidence\n" + "\n".join(lines),
        source_ids=tuple(i.item_id for i in items),
        role=TrustedRole.SYSTEM,
    )


def _tool_results_segment(
    results: tuple[ToolResult, ...], config: CompilerConfig
) -> ContextSegment:
    lines = [f"- {r.invocation_id}: {r.status.value}\n{r.content}" for r in results]
    return ContextSegment.create(
        segment_id="tool-results",
        kind="tool_results",
        priority=config.tool_results_priority,
        content="Recent tool results\n" + "\n".join(lines),
        source_ids=tuple(r.result_id for r in results),
        role=TrustedRole.TOOL,
    )


def _candidate_preview_segment(
    candidates: tuple[CandidateItem, ...], config: CompilerConfig
) -> ContextSegment:
    lines = [f"- {c.item_id}: {c.content[:200]}" for c in candidates]
    return ContextSegment.create(
        segment_id="candidate-previews",
        kind="candidate_previews",
        priority=config.candidate_preview_priority,
        content="Candidate previews\n" + "\n".join(lines),
        source_ids=tuple(c.item_id for c in candidates),
        role=TrustedRole.USER,
    )


def _contradiction_segments(
    snapshot: HarnessSnapshot, config: CompilerConfig
) -> list[ContextSegment]:
    segments: list[ContextSegment] = []
    contradictory_links = [
        link for link in snapshot.evidence_links if link.relationship == "contradicts"
    ]
    if contradictory_links:
        lines = []
        for link in contradictory_links:
            claim = next((c for c in snapshot.claims if c.claim_id == link.claim_id), None)
            item = next(
                (c for c in snapshot.curated_items if c.item_id == link.curated_item_id), None
            )
            claim_text = claim.content if claim else "unknown claim"
            item_text = item.content if item else "unknown evidence"
            lines.append(
                f"- Claim '{claim_text}' is contradicted by evidence '{item_text}'"
            )
        segments.append(
            ContextSegment.create(
                segment_id="contradictions",
                kind="contradiction_notes",
                priority=config.contradiction_notes_priority,
                content="Contradictions\n" + "\n".join(lines),
                role=TrustedRole.SYSTEM,
            )
        )
    # Also include withdrawn claims as failure notes.
    withdrawn = [c for c in snapshot.claims if c.status == ClaimStatus.WITHDRAWN]
    if withdrawn:
        lines = [f"- Withdrawn claim: {c.content}" for c in withdrawn]
        segments.append(
            ContextSegment.create(
                segment_id="withdrawn-claims",
                kind="contradiction_notes",
                priority=config.contradiction_notes_priority,
                content="Withdrawn claims\n" + "\n".join(lines),
                role=TrustedRole.SYSTEM,
            )
        )
    return segments


def _verified_curated_item_ids(snapshot: HarnessSnapshot) -> set[str]:
    verified_claim_ids = {c.claim_id for c in snapshot.claims if c.status == ClaimStatus.VERIFIED}
    return {
        link.curated_item_id
        for link in snapshot.evidence_links
        if link.claim_id in verified_claim_ids and link.relationship == "supports"
    }


# ---------------------------------------------------------------------------
# Token allocation and truncation
# ---------------------------------------------------------------------------


_BUDGET_FIELDS: dict[str, str] = {
    "task_constraints": "task_constraints_budget",
    "plan": "plan_budget",
    "verified_evidence": "verified_evidence_budget",
    "active_evidence": "active_evidence_budget",
    "contradiction_notes": "contradiction_notes_budget",
    "tool_results": "tool_results_budget",
    "candidate_previews": "candidate_preview_budget",
    "retrieval_handles": "retrieval_handles_budget",
}


def _allocate_tokens(
    segments: list[ContextSegment],
    config: CompilerConfig,
    counter: TokenCounter,
) -> list[ContextSegment]:
    # Count tokens for every segment first.
    counted = [segment.with_token_count(counter.count(segment.content)) for segment in segments]

    # Apply per-section budgets.
    section_usage: dict[str, int] = {}
    kept: list[ContextSegment] = []
    for segment in counted:
        field = _BUDGET_FIELDS.get(segment.kind)
        budget = getattr(config, field) if field else None
        section_usage.setdefault(segment.kind, 0)
        if budget is not None and section_usage[segment.kind] + segment.token_count > budget:
            continue
        section_usage[segment.kind] += segment.token_count
        kept.append(segment)

    # Global budget: sort by priority descending, then by original order as tie-breaker.
    # Evict lowest-priority segments until we fit.
    indexed = list(enumerate(kept))
    indexed.sort(key=lambda pair: (-pair[1].priority, pair[0]))

    current_total = sum(s.token_count for _, s in indexed)
    evicted_positions: set[int] = set()
    for original_index, segment in reversed(indexed):
        if current_total <= config.max_total_tokens:
            break
        evicted_positions.add(original_index)
        current_total -= segment.token_count

    # Rebuild in original order, omitting evicted segments.
    return [s for i, s in enumerate(kept) if i not in evicted_positions]


def _add_retrieval_handles(
    segments: list[ContextSegment],
    snapshot: HarnessSnapshot,
    config: CompilerConfig,
    counter: TokenCounter,
) -> list[ContextSegment]:
    # Identify source objects that did not make it into the kept segments.
    kept_source_ids = {sid for s in segments for sid in s.source_ids}
    omitted: list[tuple[str, str]] = []
    for item in snapshot.curated_items:
        if item.item_id not in kept_source_ids:
            omitted.append((item.item_id, item.content_hash))
    for result in snapshot.tool_results:
        if result.result_id not in kept_source_ids:
            omitted.append((result.result_id, result.content_hash))
    if not omitted:
        return segments

    lines = [f"- {obj_id} ({content_hash})" for obj_id, content_hash in omitted]
    handle = ContextSegment.create(
        segment_id="retrieval-handles",
        kind="retrieval_handles",
        priority=config.retrieval_handles_priority,
        content="Retrieval handles for omitted material\n" + "\n".join(lines),
        source_ids=tuple(oid for oid, _ in omitted),
    )
    handle = handle.with_token_count(counter.count(handle.content))

    # Respect retrieval handle budget and global budget.
    current_total = sum(s.token_count for s in segments)
    handle_budget = config.retrieval_handles_budget
    if handle_budget is not None and handle.token_count > handle_budget:
        return segments
    if current_total + handle.token_count > config.max_total_tokens:
        return segments
    return list(segments) + [handle]


# ---------------------------------------------------------------------------
# Epoch / prefix reuse
# ---------------------------------------------------------------------------


def compute_epoch(
    previous_packet: ContextPacket,
    current_packet: ContextPacket,
    cache_branch_id: str,
    previous_token_ids: tuple[int, ...] | None = None,
    current_token_ids: tuple[int, ...] | None = None,
) -> ContextEpoch:
    """Compare two packets and compute the prefix/suffix reuse boundary."""
    if previous_token_ids is not None and current_token_ids is not None:
        prefix_len = 0
        for a, b in zip(previous_token_ids, current_token_ids):
            if a == b:
                prefix_len += 1
            else:
                break
        prefix_tokens = prefix_len
        invalidated_tokens = len(current_token_ids) - prefix_len
    else:
        prefix_tokens = 0
        invalidated_tokens = current_packet.total_tokens
        for prev, curr in zip(previous_packet.segments, current_packet.segments):
            if prev.content_hash != curr.content_hash:
                break
            prefix_tokens += curr.token_count
        invalidated_tokens = current_packet.total_tokens - prefix_tokens
    return ContextEpoch(
        epoch_id=current_packet.epoch_id,
        parent_epoch_id=previous_packet.epoch_id,
        packet_hash=current_packet.packet_hash,
        common_prefix_tokens=prefix_tokens,
        invalidated_suffix_tokens=invalidated_tokens,
        cache_branch_id=cache_branch_id,
    )


def _compute_epoch(previous_packet: ContextPacket, packet: ContextPacket) -> ContextEpoch:
    # Side-effect-free helper used by compile_context.
    return compute_epoch(previous_packet, packet, cache_branch_id=packet.packet_hash[:16])
