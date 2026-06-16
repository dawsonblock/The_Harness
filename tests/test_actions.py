"""Tests for typed actions and validation."""

from __future__ import annotations

import re

import pytest

from rfsn_agent.actions import (
    BudgetError,
    CurateAction,
    DecomposeAction,
    PreconditionError,
    ReadAction,
    ReviseClaimAction,
    SafetyError,
    SearchAction,
    SubmitAction,
    VerifyAction,
    plan_events,
    validate_action,
)
from rfsn_agent.domain import (
    BudgetLedger,
    CandidateItem,
    Claim,
    CuratedItem,
    EvidenceLink,
    HarnessSnapshot,
    TaskNode,
)
from rfsn_agent.types import VerificationResult


def _snapshot(
    *,
    tasks: tuple[TaskNode, ...] = (),
    candidates: tuple[CandidateItem, ...] = (),
    curated_items: tuple[CuratedItem, ...] = (),
    claims: tuple[Claim, ...] = (),
    evidence_links: tuple[EvidenceLink, ...] = (),
    budget: BudgetLedger | None = None,
) -> HarnessSnapshot:
    return HarnessSnapshot.create(
        trajectory_id="traj-1",
        epoch_id="epoch-0",
        sequence=0,
        tasks=tasks,
        candidates=candidates,
        curated_items=curated_items,
        claims=claims,
        evidence_links=evidence_links,
        budget=budget,
    )


def test_validate_decompose_ok() -> None:
    snap = _snapshot()
    action = DecomposeAction(
        task_id="task-1",
        parent_task_id=None,
        description="do it",
        dependency_ids=(),
    )
    validate_action(action, snap)


def test_validate_decompose_duplicate_task_fails() -> None:
    snap = _snapshot(
        tasks=(
            TaskNode(
                task_id="task-1",
                trajectory_id="traj-1",
                parent_id=None,
                description="existing",
            ),
        ),
    )
    action = DecomposeAction(
        task_id="task-1",
        parent_task_id=None,
        description="new",
        dependency_ids=(),
    )
    with pytest.raises(PreconditionError, match="already exists"):
        validate_action(action, snap)


def test_validate_search_empty_query_fails() -> None:
    snap = _snapshot()
    with pytest.raises(SafetyError, match="non-empty"):
        validate_action(SearchAction(query="  "), snap)


def test_validate_read_forbidden_path() -> None:
    snap = _snapshot()
    pattern = re.compile(r"\.\.|")
    pattern = re.compile(r"\.\.")
    action = ReadAction(source_id="../etc/passwd")
    with pytest.raises(SafetyError, match="forbidden path"):
        validate_action(action, snap, forbidden_path_pattern=pattern)


def test_validate_curate_missing_candidate_fails() -> None:
    snap = _snapshot()
    action = CurateAction(
        candidate_ids=("cand-1",),
        curated_item_id="cur-1",
        content="body",
        priority=1,
        source_ids=(),
    )
    with pytest.raises(PreconditionError, match="unknown candidates"):
        validate_action(action, snap)


def test_validate_budget_exceeded() -> None:
    snap = _snapshot(
        budget=BudgetLedger(trajectory_id="traj-1", max_tokens=10, tokens_used=10),
    )
    action = SearchAction(query="foo")
    with pytest.raises(BudgetError, match="Token budget exceeded"):
        validate_action(action, snap, token_cost_estimate=1)


def test_validate_tool_call_budget() -> None:
    snap = _snapshot(
        budget=BudgetLedger(
            trajectory_id="traj-1",
            max_tokens=100,
            max_tool_calls=1,
            tool_calls_used=1,
        ),
    )
    action = SearchAction(query="foo")
    with pytest.raises(BudgetError, match="Tool call budget exceeded"):
        validate_action(action, snap)


def test_plan_decompose_event() -> None:
    snap = _snapshot()
    action = DecomposeAction(
        task_id="task-1",
        parent_task_id=None,
        description="do it",
        dependency_ids=(),
    )
    events = plan_events(action, snap, action_id="act-1")
    assert len(events) == 1
    assert events[0].event_type == "task_decomposed"
    assert events[0].sequence == 1


def test_plan_search_event() -> None:
    snap = _snapshot()
    action = SearchAction(query="foo")
    events = plan_events(action, snap, action_id="act-1")
    assert events[0].event_type == "tool_invoked"
    assert events[0].header.action_id == "act-1"


def test_plan_submit_event() -> None:
    snap = _snapshot()
    action = SubmitAction(
        submission_id="sub-1", content="answer", source_ids=("src-1",)
    )
    events = plan_events(action, snap, action_id="act-1")
    assert events[0].event_type == "submission_recorded"


def test_validate_revise_claim_no_change_fails() -> None:
    snap = _snapshot(
        claims=(
            Claim.create(
                claim_id="claim-1",
                trajectory_id="traj-1",
                content="foo",
            ),
        ),
    )
    action = ReviseClaimAction(claim_id="claim-1", new_content=None, new_status=None)
    with pytest.raises(PreconditionError, match="must change content or status"):
        validate_action(action, snap)


def test_validate_verify_unknown_link_fails() -> None:
    snap = _snapshot()
    action = VerifyAction(
        link_id="link-1",
        verification_id="ver-1",
        result=VerificationResult.CONFIRMED,
        details="ok",
    )
    with pytest.raises(PreconditionError, match="does not exist"):
        validate_action(action, snap)
