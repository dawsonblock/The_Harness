"""Strong identifiers and enumerations for the KV-cache domain."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import NewType

PageId = NewType("PageId", str)
NodeId = NewType("NodeId", str)
BranchId = NewType("BranchId", str)
LayerIndex = NewType("LayerIndex", int)
ContentHash = NewType("ContentHash", str)


class KVPageStatus(str, Enum):
    """Lifecycle status of a KV page."""

    COMPRESSED = "compressed"
    DECOMPRESSED = "decompressed"
    EVICTED = "evicted"


class EvictionState(str, Enum):
    """Residency state of a page in the cache."""

    PINNED = "pinned"
    RESIDENT = "resident"
    EVICTED = "evicted"


@dataclass(frozen=True, slots=True)
class ContentReference:
    """A pointer to content stored in an external content-addressed store.

    Attributes:
        content_hash: SHA-256 hash of the referenced content.
        byte_length: Size of the content in bytes.
    """

    content_hash: ContentHash
    byte_length: int
