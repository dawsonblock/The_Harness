"""Objective evaluation and cryptographic receipts for submissions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from rfsn_agent.common import canonical_json, hash_content, utc_now
from rfsn_agent.domain import SubmissionRecord
from rfsn_agent.types import ContentHash, SubmissionId, TrajectoryId


@dataclass(frozen=True, slots=True)
class ObjectiveSpec:
    """Expected output and budget constraints for a task objective."""

    objective_id: str
    expected_content: str
    max_tokens: int
    required_source_ids: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Result of comparing a submission to an objective."""

    objective_id: str
    submission_id: SubmissionId
    trajectory_id: TrajectoryId
    passed: bool
    status: Literal["passed", "failed", "inconclusive"]
    score: float
    details: str
    details_hash: ContentHash
    evaluated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        expected = hash_content(self.details)
        if expected != self.details_hash:
            raise ValueError(
                f"EvaluationResult {self.submission_id}: details_hash mismatch"
            )


@dataclass(frozen=True, slots=True)
class ObjectiveReceipt:
    """Cryptographic receipt for an objective evaluation."""

    receipt_id: str
    objective_id: str
    submission_id: SubmissionId
    trajectory_id: TrajectoryId
    submission_hash: ContentHash
    expected_hash: ContentHash
    budget_limit: int
    required_source_ids: tuple[str, ...]
    evaluation_hash: ContentHash
    status: Literal["passed", "failed", "inconclusive"]
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if self.evaluation_hash:
            expected = self.compute_hash()
            if expected != self.evaluation_hash:
                raise ValueError("ObjectiveReceipt: evaluation_hash mismatch")

    @classmethod
    def create(
        cls,
        *,
        receipt_id: str,
        objective_id: str,
        submission_id: SubmissionId,
        trajectory_id: TrajectoryId,
        submission_hash: ContentHash,
        expected_hash: ContentHash,
        budget_limit: int,
        required_source_ids: tuple[str, ...],
        status: Literal["passed", "failed", "inconclusive"],
        created_at: datetime,
    ) -> ObjectiveReceipt:
        """Create a receipt with its cryptographic hash populated."""
        transient = cls(
            receipt_id=receipt_id,
            objective_id=objective_id,
            submission_id=submission_id,
            trajectory_id=trajectory_id,
            submission_hash=submission_hash,
            expected_hash=expected_hash,
            budget_limit=budget_limit,
            required_source_ids=required_source_ids,
            evaluation_hash=ContentHash(""),
            status=status,
            created_at=created_at,
        )
        object.__setattr__(transient, "evaluation_hash", transient.compute_hash())
        return transient

    def compute_hash(self) -> ContentHash:
        payload = {
            "receipt_id": self.receipt_id,
            "objective_id": self.objective_id,
            "submission_id": self.submission_id,
            "trajectory_id": self.trajectory_id,
            "submission_hash": self.submission_hash,
            "expected_hash": self.expected_hash,
            "budget_limit": self.budget_limit,
            "required_source_ids": sorted(self.required_source_ids),
            "status": self.status,
            "created_at": self.created_at.isoformat(),
        }
        return ContentHash(hash_content(canonical_json(payload)))


class ObjectiveEvaluator:
    """Simple deterministic evaluator for text submissions."""

    def evaluate(
        self,
        *,
        submission: SubmissionRecord,
        objective: ObjectiveSpec,
        receipt_id: str | None = None,
    ) -> tuple[EvaluationResult, ObjectiveReceipt]:
        """Evaluate a submission and return a result plus a signed receipt."""
        normalized = submission.content.strip().lower()
        expected = objective.expected_content.strip().lower()
        passed = normalized == expected
        missing_sources = [
            source_id
            for source_id in objective.required_source_ids
            if source_id not in submission.source_ids
        ]
        if passed and missing_sources:
            status: Literal["passed", "failed", "inconclusive"] = "inconclusive"
            score = 0.75
            details = f"Content matches but missing sources: {sorted(missing_sources)}"
        elif passed:
            status = "passed"
            score = 1.0
            details = "Submission matches expected content."
        else:
            status = "failed"
            score = 0.0
            details = "Submission does not match expected content."

        result = EvaluationResult(
            objective_id=objective.objective_id,
            submission_id=submission.submission_id,
            trajectory_id=submission.trajectory_id,
            passed=passed,
            status=status,
            score=score,
            details=details,
            details_hash=ContentHash(hash_content(details)),
        )
        receipt = ObjectiveReceipt.create(
            receipt_id=receipt_id or f"eval-{submission.submission_id}",
            objective_id=objective.objective_id,
            submission_id=submission.submission_id,
            trajectory_id=submission.trajectory_id,
            submission_hash=submission.content_hash,
            expected_hash=ContentHash(hash_content(expected)),
            budget_limit=objective.max_tokens,
            required_source_ids=objective.required_source_ids,
            status=status,
            created_at=result.evaluated_at,
        )
        return result, receipt


__all__: list[str] = [
    "EvaluationResult",
    "ObjectiveEvaluator",
    "ObjectiveReceipt",
    "ObjectiveSpec",
]
