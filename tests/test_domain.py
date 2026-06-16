"""Tests for immutable domain schemas."""

from __future__ import annotations

import dataclasses

import pytest

from rfsn_agent.common import canonical_json, hash_content
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
from rfsn_agent.types import (
    ClaimStatus,
    TaskStatus,
    ToolStatus,
    VerificationResult,
)


def test_candidate_item_create_and_immutable() -> None:
    item = CandidateItem.create(
        item_id="cand-1",
        trajectory_id="traj-1",
        source_id="src-1",
        retrieval_query="foo bar",
        content="candidate content",
        metadata=(("url", "http://example.com"),),
        provenance=Provenance(actor="policy", action_id="search-1"),
    )
    assert item.content_hash == hash_content("candidate content")
    assert item.provenance.actor == "policy"

    with pytest.raises(dataclasses.FrozenInstanceError):
        item.content = "mutated"  # type: ignore[misc]


def test_candidate_item_rejects_bad_hash() -> None:
    with pytest.raises(ValueError, match="content_hash mismatch"):
        CandidateItem(
            item_id="cand-1",
            trajectory_id="traj-1",
            source_id="src-1",
            retrieval_query="foo bar",
            content="candidate content",
            content_hash="badhash",
        )


def test_curated_item_links_candidates() -> None:
    curated = CuratedItem.create(
        item_id="cur-1",
        trajectory_id="traj-1",
        candidate_ids=("cand-1", "cand-2"),
        content="curated content",
        priority=5,
        source_ids=("src-1",),
    )
    assert curated.candidate_ids == ("cand-1", "cand-2")
    assert curated.priority == 5
    assert curated.content_hash == hash_content("curated content")


def test_claim_status_transition() -> None:
    claim = Claim.create(
        claim_id="claim-1",
        trajectory_id="traj-1",
        content="the sky is blue",
    )
    assert claim.status == ClaimStatus.STATED

    verified = claim.with_status(
        ClaimStatus.VERIFIED,
        Provenance(actor="verifier", action_id="verify-1"),
    )
    assert verified.status == ClaimStatus.VERIFIED
    assert verified.provenance.actor == "verifier"
    # Original unchanged.
    assert claim.status == ClaimStatus.STATED


def test_evidence_link_strength_bounds() -> None:
    EvidenceLink(
        link_id="link-1",
        trajectory_id="traj-1",
        claim_id="claim-1",
        curated_item_id="cur-1",
        relationship="supports",
        strength=0.8,
    )

    with pytest.raises(ValueError, match="strength must be in"):
        EvidenceLink(
            link_id="link-2",
            trajectory_id="traj-1",
            claim_id="claim-1",
            curated_item_id="cur-1",
            relationship="supports",
            strength=1.5,
        )


def test_verification_record_hash() -> None:
    record = VerificationRecord.create(
        record_id="ver-1",
        trajectory_id="traj-1",
        claim_id="claim-1",
        method="web_search",
        result=VerificationResult.CONFIRMED,
        details="confirmed by two sources",
    )
    assert record.details_hash == hash_content("confirmed by two sources")

    with pytest.raises(ValueError, match="details_hash mismatch"):
        VerificationRecord(
            record_id="ver-2",
            trajectory_id="traj-1",
            claim_id="claim-1",
            method="web_search",
            result=VerificationResult.INCONCLUSIVE,
            details="details",
            details_hash="badhash",
        )


def test_task_node_defaults() -> None:
    task = TaskNode(
        task_id="task-1",
        trajectory_id="traj-1",
        parent_id=None,
        description="do something",
    )
    assert task.status == TaskStatus.PENDING
    assert task.dependency_ids == ()


def test_budget_ledger_reserve_and_spend() -> None:
    budget = BudgetLedger(trajectory_id="traj-1", max_tokens=1000)
    assert budget.tokens_available == 1000

    reserved = budget.reserve(200)
    assert reserved.tokens_reserved == 200
    assert reserved.tokens_available == 800

    spent = reserved.spend(tokens=150, tool_calls=1, wall_seconds=2.0)
    assert spent.tokens_used == 150
    assert spent.tokens_reserved == 50
    assert spent.tool_calls_used == 1
    assert spent.wall_seconds_used == 2.0
    assert spent.tokens_available == 800

    with pytest.raises(ValueError, match="Cannot spend negative"):
        budget.spend(tokens=-1)


def test_tool_invocation_arguments_hash() -> None:
    invocation = ToolInvocation.create(
        invocation_id="tool-1",
        trajectory_id="traj-1",
        action_id="action-1",
        parent_task_id="task-1",
        tool_name="read_file",
        arguments=(("path", "/tmp/foo"),),
        dependency_ids=("tool-0",),
    )
    assert invocation.arguments_hash == hash_content(
        canonical_json({"path": "/tmp/foo"})
    )


def test_tool_result_hash() -> None:
    result = ToolResult.create(
        result_id="res-1",
        invocation_id="tool-1",
        trajectory_id="traj-1",
        status=ToolStatus.SUCCESS,
        content="file contents",
    )
    assert result.content_hash == hash_content("file contents")


def test_submission_record_source_ids() -> None:
    sub = SubmissionRecord.create(
        submission_id="sub-1",
        trajectory_id="traj-1",
        content="final answer",
        source_ids=("src-1", "src-2"),
    )
    assert sub.source_ids == ("src-1", "src-2")


def test_harness_snapshot_state_hash() -> None:
    candidate = CandidateItem.create(
        item_id="cand-1",
        trajectory_id="traj-1",
        source_id="src-1",
        retrieval_query="q",
        content="candidate",
    )
    snapshot = HarnessSnapshot.create(
        trajectory_id="traj-1",
        epoch_id="epoch-1",
        sequence=0,
        candidates=(candidate,),
        budget=BudgetLedger(trajectory_id="traj-1", max_tokens=1000),
    )
    assert snapshot.state_hash == snapshot.compute_state_hash()
    assert snapshot.state_hash != ""

    with pytest.raises(ValueError, match="state_hash mismatch"):
        HarnessSnapshot(
            trajectory_id="traj-1",
            epoch_id="epoch-1",
            sequence=0,
            state_hash="badhash",
        )


def test_harness_snapshot_deterministic_hash() -> None:
    """Same inputs must produce identical state hashes."""
    candidate = CandidateItem.create(
        item_id="cand-1",
        trajectory_id="traj-1",
        source_id="src-1",
        retrieval_query="q",
        content="candidate",
    )
    s1 = HarnessSnapshot.create(
        trajectory_id="traj-1",
        epoch_id="epoch-1",
        sequence=0,
        candidates=(candidate,),
    )
    s2 = HarnessSnapshot.create(
        trajectory_id="traj-1",
        epoch_id="epoch-1",
        sequence=0,
        candidates=(candidate,),
    )
    assert s1.state_hash == s2.state_hash


def test_provenance_with_event() -> None:
    prov = Provenance(actor="policy", action_id="act-1")
    stamped = prov.with_event("evt-1")
    assert stamped.event_id == "evt-1"
    assert prov.event_id is None
