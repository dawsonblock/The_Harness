"""Tests for objective evaluation and receipts."""

from __future__ import annotations

import pytest

from rfsn_agent.domain import SubmissionRecord
from rfsn_agent.evaluation import ObjectiveEvaluator, ObjectiveSpec


def _submission(content: str, source_ids: tuple[str, ...] = ("src-1",)) -> SubmissionRecord:
    return SubmissionRecord.create(
        submission_id="sub-1",
        trajectory_id="traj-1",
        content=content,
        source_ids=source_ids,
    )


def test_objective_evaluator_passes_matching_submission() -> None:
    evaluator = ObjectiveEvaluator()
    result, receipt = evaluator.evaluate(
        submission=_submission("final answer"),
        objective=ObjectiveSpec(
            objective_id="obj-1",
            expected_content="final answer",
            max_tokens=100,
            required_source_ids=("src-1",),
        ),
        receipt_id="receipt-1",
    )
    assert result.status == "passed"
    assert result.score == 1.0
    assert result.details_hash is not None
    assert receipt.receipt_id == "receipt-1"
    assert receipt.submission_hash == _submission("final answer").content_hash
    assert receipt.expected_hash is not None
    assert receipt.status == "passed"


def test_objective_evaluator_marks_missing_sources_inconclusive() -> None:
    evaluator = ObjectiveEvaluator()
    result, receipt = evaluator.evaluate(
        submission=_submission("final answer"),
        objective=ObjectiveSpec(
            objective_id="obj-2",
            expected_content="final answer",
            max_tokens=100,
            required_source_ids=("src-1", "src-2"),
        ),
    )
    assert result.status == "inconclusive"
    assert result.score == 0.75
    assert receipt.status == "inconclusive"


def test_objective_evaluator_fails_mismatched_submission() -> None:
    evaluator = ObjectiveEvaluator()
    result, receipt = evaluator.evaluate(
        submission=_submission("wrong answer"),
        objective=ObjectiveSpec(
            objective_id="obj-3",
            expected_content="final answer",
            max_tokens=100,
        ),
    )
    assert result.status == "failed"
    assert result.score == 0.0
    assert receipt.status == "failed"


def test_objective_receipt_rejects_hash_mismatch() -> None:
    with pytest.raises(ValueError, match="evaluation_hash mismatch"):
        from rfsn_agent.evaluation import ObjectiveReceipt
        from rfsn_agent.types import ContentHash, SubmissionId, TrajectoryId

        ObjectiveReceipt(
            receipt_id="receipt-bad",
            objective_id="obj-1",
            submission_id=SubmissionId("sub-1"),
            trajectory_id=TrajectoryId("traj-1"),
            submission_hash=ContentHash("a" * 64),
            expected_hash=ContentHash("b" * 64),
            budget_limit=100,
            required_source_ids=("src-1",),
            evaluation_hash=ContentHash("0" * 64),
            status="passed",
        )
