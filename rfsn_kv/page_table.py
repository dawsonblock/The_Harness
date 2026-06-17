"""Immutable page table mapping logical token positions to physical pages.

The page table is the core data structure that bridges the logical KV sequence
(order of tokens) to the physical storage (pages on disk or in memory). It is
immutable: every mutation returns a new PageTable instance.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from rfsn_kv.common import canonical_json, hash_content
from rfsn_kv.pages import PageRange
from rfsn_kv.types import ContentHash, LayerIndex, PageId


@dataclass(frozen=True, slots=True)
class PageTableEntry:
    """A single logical→physical position mapping.

    Attributes:
        logical_position: Position in the logical token sequence.
        page_id: Physical page identifier.
        token_offset: Starting token offset within the page.
        token_count: Number of tokens this entry covers in the page.
        layer_index: Transformer layer this entry belongs to.
    """

    logical_position: int
    page_id: PageId
    token_offset: int
    token_count: int
    layer_index: LayerIndex
    actor: str = "system"
    action_id: str = "init"
    event_id: str | None = None

    def __post_init__(self) -> None:
        if self.logical_position < 0:
            raise ValueError(
                f"PageTableEntry: logical_position must be >= 0, "
                f"got {self.logical_position}"
            )
        if self.token_offset < 0:
            raise ValueError(
                f"PageTableEntry: token_offset must be >= 0, "
                f"got {self.token_offset}"
            )
        if self.token_count <= 0:
            raise ValueError(
                f"PageTableEntry: token_count must be > 0, "
                f"got {self.token_count}"
            )

    @property
    def logical_end(self) -> int:
        """Exclusive end of the logical range covered by this entry."""
        return self.logical_position + self.token_count


@dataclass(frozen=True, slots=True)
class PageTable:
    """An immutable, sorted collection of page table entries.

    The page table maps logical token positions to physical page storage.
    Entries are stored sorted by logical_position and must not overlap.
    Every mutation returns a new PageTable instance.

    Attributes:
        entries: Sorted tuple of page table entries.
    """

    entries: tuple[PageTableEntry, ...] = field(default_factory=tuple)
    table_hash: ContentHash = field(default=ContentHash(""))

    def __post_init__(self) -> None:
        # Validate sorted order and no duplicates.
        for i in range(1, len(self.entries)):
            if self.entries[i].logical_position <= self.entries[i - 1].logical_position:
                raise ValueError(
                    "PageTable entries must be sorted by logical_position "
                    "with no duplicates"
                )
        # Compute hash; if table_hash was left empty, fill it in.
        expected = self._compute_hash()
        if self.table_hash != expected:
            object.__setattr__(self, "table_hash", expected)

    def _compute_hash(self) -> ContentHash:
        """Compute a deterministic hash over all entries."""
        payload = {
            "entries": [
                {
                    "logical_position": e.logical_position,
                    "page_id": e.page_id,
                    "token_offset": e.token_offset,
                    "token_count": e.token_count,
                    "layer_index": e.layer_index,
                }
                for e in self.entries
            ]
        }
        return hash_content(canonical_json(payload))

    def lookup(self, logical_position: int) -> PageTableEntry | None:
        """Find the entry containing ``logical_position``, or None."""
        # Binary search over the sorted entries.
        lo, hi = 0, len(self.entries) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            entry = self.entries[mid]
            if entry.logical_position <= logical_position < entry.logical_end:
                return entry
            elif logical_position < entry.logical_position:
                hi = mid - 1
            else:
                lo = mid + 1
        return None

    def range_query(self, start_token: int, end_token: int) -> PageRange:
        """Return the PageRange covering logical tokens [start_token, end_token)."""
        if end_token <= start_token:
            raise ValueError(
                f"range_query: end_token ({end_token}) must be > "
                f"start_token ({start_token})"
            )
        matching: list[PageId] = []
        for entry in self.entries:
            if entry.logical_end <= start_token:
                continue
            if entry.logical_position >= end_token:
                break
            matching.append(entry.page_id)
        return PageRange(
            start_token=start_token,
            end_token=end_token,
            page_ids=tuple(matching),
        )

    def with_entry(self, entry: PageTableEntry) -> PageTable:
        """Return a new table with ``entry`` inserted in sorted position."""
        # Find insertion point.
        new_entries = list(self.entries)
        pos = 0
        for i, existing in enumerate(new_entries):
            if existing.logical_position > entry.logical_position:
                pos = i
                break
        else:
            pos = len(new_entries)
        new_entries.insert(pos, entry)
        return PageTable(entries=tuple(new_entries))

    def without_entries(self, page_ids: frozenset[PageId] | set[PageId]) -> PageTable:
        """Return a new table with all entries whose page_id is in the given set removed."""
        removed = frozenset(page_ids)
        filtered = tuple(e for e in self.entries if e.page_id not in removed)
        return PageTable(entries=filtered)

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[PageTableEntry]:
        return iter(self.entries)
