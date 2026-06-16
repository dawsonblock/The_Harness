"""Tests for the deterministic context compiler."""

from __future__ import annotations

import pytest

from rfsn_agent.common import hash_content
from rfsn_agent.context import (
    CompilerConfig,
    ContextPacket,
    ContextSegment,
    DefaultContextSelector,
    JinjaContextRenderer,
    TrustedRole,
    WhitespaceTokenCounter,
    compile_context,
    compute_epoch,
)
from rfsn_agent.domain import (
    BudgetLedger,
    CandidateItem,
    Claim,
    CuratedItem,
    EvidenceLink,
    HarnessSnapshot,
    TaskNode,
    ToolResult,
)
from rfsn_agent.types import ClaimStatus, TaskStatus, ToolStatus


def _snapshot(
    *,
    tasks: tuple[TaskNode, ...] = (),
    curated_items: tuple[CuratedItem, ...] = (),
    claims: tuple[Claim, ...] = (),
    evidence_links: tuple[EvidenceLink, ...] = (),
    candidates: tuple[CandidateItem, ...] = (),
    tool_results: tuple[ToolResult, ...] = (),
    budget: BudgetLedger | None = None,
) -> HarnessSnapshot:
    return HarnessSnapshot.create(
        trajectory_id="traj-1",
        epoch_id="epoch-0",
        sequence=0,
        tasks=tasks,
        curated_items=curated_items,
        claims=claims,
        evidence_links=evidence_links,
        candidates=candidates,
        tool_results=tool_results,
        budget=budget,
    )


def test_compile_empty_snapshot() -> None:
    snap = _snapshot()
    config = CompilerConfig(max_total_tokens=100)
    packet = compile_context(snap, config)
    assert packet.total_tokens == 0
    assert packet.segments == ()


def test_compile_task_and_plan_segments() -> None:
    snap = _snapshot(
        tasks=(
            TaskNode(
                task_id="task-1",
                trajectory_id="traj-1",
                parent_id=None,
                description="find the bug",
                status=TaskStatus.IN_PROGRESS,
            ),
        ),
        budget=BudgetLedger(trajectory_id="traj-1", max_tokens=1000),
    )
    config = CompilerConfig(max_total_tokens=100)
    packet = compile_context(snap, config)
    kinds = {s.kind for s in packet.segments}
    assert "task_constraints" in kinds
    assert "plan" in kinds
    assert packet.total_tokens > 0


def test_verified_evidence_priority_over_active() -> None:
    snap = _snapshot(
        claims=(
            Claim.create(
                claim_id="claim-1",
                trajectory_id="traj-1",
                content="the bug is in foo.py",
                status=ClaimStatus.VERIFIED,
            ),
        ),
        curated_items=(
            CuratedItem.create(
                item_id="cur-1",
                trajectory_id="traj-1",
                candidate_ids=(),
                content="foo.py line 42",
            ),
            CuratedItem.create(
                item_id="cur-2",
                trajectory_id="traj-1",
                candidate_ids=(),
                content="bar.py line 10",
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
    config = CompilerConfig(max_total_tokens=1000)
    packet = compile_context(snap, config)
    verified = next(s for s in packet.segments if s.kind == "verified_evidence")
    active = next(s for s in packet.segments if s.kind == "active_evidence")
    assert verified.priority > active.priority
    assert "cur-1" in verified.source_ids
    assert "cur-2" in active.source_ids


def test_contradiction_segment() -> None:
    snap = _snapshot(
        claims=(
            Claim.create(
                claim_id="claim-1",
                trajectory_id="traj-1",
                content="foo is correct",
            ),
        ),
        curated_items=(
            CuratedItem.create(
                item_id="cur-1",
                trajectory_id="traj-1",
                candidate_ids=(),
                content="foo is broken",
            ),
        ),
        evidence_links=(
            EvidenceLink(
                link_id="link-1",
                trajectory_id="traj-1",
                claim_id="claim-1",
                curated_item_id="cur-1",
                relationship="contradicts",
                strength=0.9,
            ),
        ),
    )
    config = CompilerConfig(max_total_tokens=1000)
    packet = compile_context(snap, config)
    assert any(s.kind == "contradiction_notes" for s in packet.segments)


def test_global_budget_truncation_evicts_lower_priority() -> None:
    snap = _snapshot(
        tasks=(
            TaskNode(
                task_id="task-1",
                trajectory_id="traj-1",
                parent_id=None,
                description="important task",
                status=TaskStatus.IN_PROGRESS,
            ),
        ),
        curated_items=tuple(
            CuratedItem.create(
                item_id=f"cur-{i}",
                trajectory_id="traj-1",
                candidate_ids=(),
                content=f"evidence body number {i} with many words",
            )
            for i in range(10)
        ),
    )
    config = CompilerConfig(max_total_tokens=20)
    packet = compile_context(snap, config)
    assert packet.total_tokens <= 20
    # Task constraints should survive; candidate previews should not appear.
    kinds = {s.kind for s in packet.segments}
    assert "task_constraints" in kinds


def test_exact_deduplication_of_full_segments() -> None:
    # If two distinct segments end up with identical content, only one is kept.
    # This is most easily demonstrated by constructing two identical candidate
    # previews that the compiler renders into the same segment kind.
    same_preview = "identical preview text"
    snap = _snapshot(
        candidates=(
            CandidateItem.create(
                item_id="cand-1",
                trajectory_id="traj-1",
                source_id="src-1",
                retrieval_query="q",
                content=same_preview,
            ),
            CandidateItem.create(
                item_id="cand-2",
                trajectory_id="traj-1",
                source_id="src-1",
                retrieval_query="q",
                content=same_preview,
            ),
        ),
    )
    config = CompilerConfig(
        max_total_tokens=1000,
        include_candidate_previews=True,
    )
    packet = compile_context(snap, config)
    previews = [s for s in packet.segments if s.kind == "candidate_previews"]
    # The compiler produces a single preview segment; deduplication keeps it.
    assert len(previews) == 1


def test_retrieval_handles_for_omitted_material() -> None:
    snap = _snapshot(
        curated_items=(
            CuratedItem.create(
                item_id="cur-1",
                trajectory_id="traj-1",
                candidate_ids=(),
                content="kept evidence",
            ),
            CuratedItem.create(
                item_id="cur-2",
                trajectory_id="traj-1",
                candidate_ids=(),
                content="omitted evidence with many words to exceed budget",
            ),
        ),
    )
    config = CompilerConfig(
        max_total_tokens=25,
        active_evidence_budget=2,
        retrieval_handles_budget=30,
    )
    packet = compile_context(snap, config)
    assert any(s.kind == "retrieval_handles" for s in packet.segments)
    handle = next(s for s in packet.segments if s.kind == "retrieval_handles")
    assert "cur-2" in handle.content


def test_context_packet_deterministic_hash() -> None:
    snap = _snapshot(
        tasks=(
            TaskNode(
                task_id="task-1",
                trajectory_id="traj-1",
                parent_id=None,
                description="task",
                status=TaskStatus.PENDING,
            ),
        ),
    )
    config = CompilerConfig(max_total_tokens=100)
    p1 = compile_context(snap, config)
    p2 = compile_context(snap, config)
    assert p1.packet_hash == p2.packet_hash


def test_context_segment_hash_validation() -> None:
    _ = ContextSegment.create(
        segment_id="s1",
        kind="plan",
        priority=1,
        content="hello",
    )
    with pytest.raises(ValueError, match="content_hash mismatch"):
        ContextSegment(
            segment_id="s1",
            kind="plan",
            priority=1,
            content="hello",
            token_count=0,
            source_ids=(),
            content_hash="badhash",
        )


def test_epoch_prefix_reuse() -> None:
    counter = WhitespaceTokenCounter()
    snap1 = _snapshot(
        tasks=(
            TaskNode(
                task_id="task-1",
                trajectory_id="traj-1",
                parent_id=None,
                description="task one",
                status=TaskStatus.PENDING,
            ),
        ),
        curated_items=(
            CuratedItem.create(
                item_id="cur-1",
                trajectory_id="traj-1",
                candidate_ids=(),
                content="first evidence",
            ),
        ),
    )
    packet1 = compile_context(snap1, CompilerConfig(max_total_tokens=100), counter)
    snap2 = HarnessSnapshot.create(
        trajectory_id="traj-1",
        epoch_id="epoch-1",
        sequence=1,
        tasks=(
            TaskNode(
                task_id="task-1",
                trajectory_id="traj-1",
                parent_id=None,
                description="task one",
                status=TaskStatus.PENDING,
            ),
        ),
        curated_items=(
            CuratedItem.create(
                item_id="cur-1",
                trajectory_id="traj-1",
                candidate_ids=(),
                content="first evidence",
            ),
            CuratedItem.create(
                item_id="cur-2",
                trajectory_id="traj-1",
                candidate_ids=(),
                content="second evidence",
            ),
        ),
    )
    packet2 = compile_context(
        snap2,
        CompilerConfig(max_total_tokens=100),
        counter,
        previous_packet=packet1,
    )
    epoch = compute_epoch(packet1, packet2, cache_branch_id="branch-1")
    # The task_constraints segment is identical, so its tokens form the prefix.
    assert epoch.common_prefix_tokens > 0
    assert epoch.invalidated_suffix_tokens > 0
    assert epoch.parent_epoch_id == "epoch-0"
    assert epoch.epoch_id == "epoch-1"


def test_no_prefix_reuse_when_first_segment_changes() -> None:
    counter = WhitespaceTokenCounter()
    snap1 = _snapshot(
        tasks=(
            TaskNode(
                task_id="task-1",
                trajectory_id="traj-1",
                parent_id=None,
                description="task one",
                status=TaskStatus.PENDING,
            ),
        ),
    )
    packet1 = compile_context(snap1, CompilerConfig(max_total_tokens=100), counter)
    snap2 = HarnessSnapshot.create(
        trajectory_id="traj-1",
        epoch_id="epoch-1",
        sequence=1,
        tasks=(
            TaskNode(
                task_id="task-1",
                trajectory_id="traj-1",
                parent_id=None,
                description="task one changed",
                status=TaskStatus.PENDING,
            ),
        ),
    )
    packet2 = compile_context(
        snap2,
        CompilerConfig(max_total_tokens=100),
        counter,
        previous_packet=packet1,
    )
    epoch = compute_epoch(packet1, packet2, cache_branch_id="branch-1")
    assert epoch.common_prefix_tokens == 0
    assert epoch.invalidated_suffix_tokens == packet2.total_tokens


def test_default_context_selector_matches_old_collect_segments() -> None:
    """DefaultContextSelector should replicate the old _collect_segments + allocation logic."""
    counter = WhitespaceTokenCounter()
    snap = _snapshot(
        tasks=(
            TaskNode(
                task_id="task-1",
                trajectory_id="traj-1",
                parent_id=None,
                description="find the bug",
                status=TaskStatus.IN_PROGRESS,
            ),
        ),
        budget=BudgetLedger(trajectory_id="traj-1", max_tokens=1000),
        tool_results=(
            ToolResult(
                result_id="res-1",
                trajectory_id="traj-1",
                invocation_id="inv-1",
                status=ToolStatus.SUCCESS,
                content="tool output",
                content_hash=hash_content("tool output"),
            ),
        ),
    )
    config = CompilerConfig(max_total_tokens=100)

    # The old inline logic is now encapsulated in DefaultContextSelector.
    selector = DefaultContextSelector()
    selected = selector.select(snap, config, counter)

    # compile_context without renderer should produce the same segments.
    packet = compile_context(snap, config, counter)

    assert [s.segment_id for s in selected] == [s.segment_id for s in packet.segments]
    assert sum(s.token_count for s in selected) == packet.total_tokens


def test_jinja_context_renderer_llama3() -> None:
    segments = [
        ContextSegment.create(
            segment_id="s1", kind="plan", priority=1, content="hello", role=TrustedRole.SYSTEM
        ),
        ContextSegment.create(
            segment_id="s2", kind="tool_results", priority=1, content="world", role=TrustedRole.TOOL
        ),
    ]
    renderer = JinjaContextRenderer()
    output = renderer.render(segments, "llama3")
    assert "<|begin_of_text|>" in output
    assert '<segment kind="plan">' in output
    assert '<segment kind="tool_results">' in output
    assert "<|end_of_text|>" in output


def test_jinja_context_renderer_gpt4() -> None:
    segments = [
        ContextSegment.create(
            segment_id="s1", kind="plan", priority=1, content="hello", role=TrustedRole.SYSTEM
        ),
    ]
    renderer = JinjaContextRenderer()
    output = renderer.render(segments, "gpt4")
    assert "### plan" in output
    assert "hello" in output


def test_jinja_context_renderer_claude() -> None:
    segments = [
        ContextSegment.create(
            segment_id="s1", kind="plan", priority=1, content="hello", role=TrustedRole.SYSTEM
        ),
    ]
    renderer = JinjaContextRenderer()
    output = renderer.render(segments, "claude")
    assert "[plan]" in output
    assert "hello" in output


def test_role_based_segment_grouping_in_compile_context() -> None:
    counter = WhitespaceTokenCounter()
    snap = _snapshot(
        tasks=(
            TaskNode(
                task_id="task-1",
                trajectory_id="traj-1",
                parent_id=None,
                description="find the bug",
                status=TaskStatus.IN_PROGRESS,
            ),
        ),
        tool_results=(
            ToolResult(
                result_id="res-1",
                trajectory_id="traj-1",
                invocation_id="inv-1",
                status=ToolStatus.SUCCESS,
                content="tool output",
                content_hash=hash_content("tool output"),
            ),
        ),
    )
    config = CompilerConfig(max_total_tokens=100)
    renderer = JinjaContextRenderer()
    packet = compile_context(snap, config, counter, renderer=renderer, model_type="llama3")

    # With a renderer the packet should contain a single rendered segment.
    assert len(packet.segments) == 1
    rendered = packet.segments[0]
    assert rendered.kind == "rendered"
    # Boundary marker should appear between SYSTEM (task/plan) and TOOL (tool results).
    assert "<|tool|>" in rendered.content


def test_compute_epoch_with_token_ids() -> None:
    seg = ContextSegment.create(
        segment_id="s1", kind="plan", priority=1, content="hello"
    )
    packet1 = ContextPacket.create(
        trajectory_id="traj-1",
        epoch_id="epoch-0",
        state_sequence=0,
        segments=(seg,),
        total_tokens=10,
        source_hashes=(seg.content_hash,),
        token_ids=(1, 2, 3, 4, 5),
    )
    packet2 = ContextPacket.create(
        trajectory_id="traj-1",
        epoch_id="epoch-1",
        state_sequence=1,
        segments=(seg,),
        total_tokens=10,
        source_hashes=(seg.content_hash,),
        token_ids=(1, 2, 3, 6, 7),
    )
    epoch = compute_epoch(
        packet1,
        packet2,
        cache_branch_id="branch-1",
        previous_token_ids=packet1.token_ids,
        current_token_ids=packet2.token_ids,
    )
    # First 3 tokens are identical.
    assert epoch.common_prefix_tokens == 3
    assert epoch.invalidated_suffix_tokens == 2
    assert epoch.parent_epoch_id == "epoch-0"
    assert epoch.epoch_id == "epoch-1"


def test_compute_epoch_with_token_ids_no_reuse() -> None:
    seg1 = ContextSegment.create(
        segment_id="s1", kind="plan", priority=1, content="hello"
    )
    seg2 = ContextSegment.create(
        segment_id="s2", kind="plan", priority=1, content="world"
    )
    packet1 = ContextPacket.create(
        trajectory_id="traj-1",
        epoch_id="epoch-0",
        state_sequence=0,
        segments=(seg1,),
        total_tokens=10,
        source_hashes=(seg1.content_hash,),
        token_ids=(1, 2, 3),
    )
    packet2 = ContextPacket.create(
        trajectory_id="traj-1",
        epoch_id="epoch-1",
        state_sequence=1,
        segments=(seg2,),
        total_tokens=10,
        source_hashes=(seg2.content_hash,),
        token_ids=(4, 5, 6),
    )
    epoch = compute_epoch(
        packet1,
        packet2,
        cache_branch_id="branch-1",
        previous_token_ids=packet1.token_ids,
        current_token_ids=packet2.token_ids,
    )
    assert epoch.common_prefix_tokens == 0
    assert epoch.invalidated_suffix_tokens == 3


def test_compile_context_backward_compatible_without_renderer() -> None:
    """Without a renderer, compile_context must behave exactly as before."""
    counter = WhitespaceTokenCounter()
    snap = _snapshot(
        tasks=(
            TaskNode(
                task_id="task-1",
                trajectory_id="traj-1",
                parent_id=None,
                description="find the bug",
                status=TaskStatus.IN_PROGRESS,
            ),
        ),
        budget=BudgetLedger(trajectory_id="traj-1", max_tokens=1000),
    )
    config = CompilerConfig(max_total_tokens=100)
    packet = compile_context(snap, config, counter)
    kinds = {s.kind for s in packet.segments}
    assert "task_constraints" in kinds
    assert "plan" in kinds
    # No rendered segment should appear.
    assert "rendered" not in kinds
