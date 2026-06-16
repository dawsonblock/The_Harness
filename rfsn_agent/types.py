"""Strong identifiers and enumerations for the harness domain."""

from __future__ import annotations

from enum import Enum
from typing import NewType

EventId = NewType("EventId", str)
TrajectoryId = NewType("TrajectoryId", str)
ItemId = NewType("ItemId", str)
TaskId = NewType("TaskId", str)
ClaimId = NewType("ClaimId", str)
LinkId = NewType("LinkId", str)
VerificationId = NewType("VerificationId", str)
ToolInvocationId = NewType("ToolInvocationId", str)
SubmissionId = NewType("SubmissionId", str)
EpochId = NewType("EpochId", str)
ContentHash = NewType("ContentHash", str)


class ClaimStatus(str, Enum):
    STATED = "stated"
    VERIFIED = "verified"
    CONTRADICTED = "contradicted"
    WITHDRAWN = "withdrawn"


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class VerificationResult(str, Enum):
    CONFIRMED = "confirmed"
    REFUTED = "refuted"
    INCONCLUSIVE = "inconclusive"


class VerificationStatus(str, Enum):
    UNVERIFIED = "unverified"
    VERIFIED = "verified"
    REFUTED = "refuted"
    INCONCLUSIVE = "inconclusive"
    STALE = "stale"


class ToolStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
