"""Residency and eviction policy for paged KV caches.

Manages which pages are currently resident in memory, enforces a maximum
resident page budget, and selects victims for eviction when the budget is
exceeded. Prefix-index shared pages are pinned and cannot be evicted.

Every mutation returns a new ``ResidencyManager`` instance (immutable
pattern consistent with the rest of the codebase).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from rfsn_kv.common import utc_now
from rfsn_kv.types import PageId


@dataclass(frozen=True, slots=True)
class ResidentPageEntry:
    """Tracks the residency state of a single page.

    Attributes:
        page_id: The page's unique identifier.
        loaded_at: When this page was loaded into the cache.
        last_accessed: When this page was last touched.
        pin_count: Number of references pinning this page (e.g., from
            prefix-index branches). Pinned pages cannot be evicted.
        actor: Entity that loaded this page.
        action_id: Action identifier for provenance tracking.
        event_id: Optional event identifier for provenance tracking.
    """

    page_id: PageId
    loaded_at: datetime = field(default_factory=utc_now)
    last_accessed: datetime = field(default_factory=utc_now)
    pin_count: int = 0
    actor: str = "system"
    action_id: str = "init"
    event_id: str | None = None

    @property
    def is_pinned(self) -> bool:
        """Return True if this page is pinned and cannot be evicted."""
        return self.pin_count > 0


class EvictionPolicy(Protocol):
    """Protocol for eviction policy implementations."""

    def select_victim(
        self, candidates: tuple[ResidentPageEntry, ...]
    ) -> ResidentPageEntry | None:
        """Select a page to evict from the given candidates.

        Returns None if no evictable page is found.
        """
        ...


class LRUEvictionPolicy:
    """Least-recently-used eviction policy.

    Selects the page with the oldest ``last_accessed`` timestamp among
    unpinned candidates.
    """

    def select_victim(
        self, candidates: tuple[ResidentPageEntry, ...]
    ) -> ResidentPageEntry | None:
        """Select the least-recently-used unpinned page."""
        evictable = [c for c in candidates if not c.is_pinned]
        if not evictable:
            return None
        return min(evictable, key=lambda e: e.last_accessed)


@dataclass(frozen=True, slots=True)
class ResidencyManager:
    """Manages which pages are resident in memory.

    Enforces a maximum resident page budget. When the budget is exceeded,
    the eviction policy selects a victim for removal. Pages pinned via the
    prefix-index (pin_count > 0) are immune to eviction.

    Every operation returns a new ``ResidencyManager`` instance.

    Attributes:
        eviction_policy: The policy used to select eviction victims.
        max_resident_pages: Maximum number of pages allowed in memory.
        resident: Tuple of currently resident page entries.
    """

    eviction_policy: EvictionPolicy = field(default_factory=LRUEvictionPolicy)
    max_resident_pages: int = 128
    resident: tuple[ResidentPageEntry, ...] = field(default_factory=tuple)

    def is_resident(self, page_id: PageId) -> bool:
        """Return True if the page is currently resident."""
        return any(e.page_id == page_id for e in self.resident)

    def resident_count(self) -> int:
        """Return the number of currently resident pages."""
        return len(self.resident)

    def get_entry(self, page_id: PageId) -> ResidentPageEntry | None:
        """Look up a resident page entry by its ID."""
        for e in self.resident:
            if e.page_id == page_id:
                return e
        return None

    def load(self, page_id: PageId) -> ResidencyManager:
        """Load a page into the cache.

        If the page is already resident, its ``last_accessed`` timestamp
        is updated. If the cache is full, eviction is triggered first.
        """
        # Already resident — just touch it.
        if self.is_resident(page_id):
            return self._touch(page_id)

        # Evict if over budget.
        manager = self
        if manager.resident_count() >= manager.max_resident_pages:
            manager, _ = manager.evict()

        entry = ResidentPageEntry(page_id=page_id)
        return ResidencyManager(
            eviction_policy=manager.eviction_policy,
            max_resident_pages=manager.max_resident_pages,
            resident=manager.resident + (entry,),
        )

    def unload(self, page_id: PageId) -> ResidencyManager:
        """Remove a page from the cache.

        If the page is pinned (pin_count > 0), the pin count is
        decremented instead of removing the page.
        """
        entry = self.get_entry(page_id)
        if entry is None:
            return self

        if entry.pin_count > 0:
            decremented = ResidentPageEntry(
                page_id=entry.page_id,
                loaded_at=entry.loaded_at,
                last_accessed=entry.last_accessed,
                pin_count=entry.pin_count - 1,
            )
            return ResidencyManager(
                eviction_policy=self.eviction_policy,
                max_resident_pages=self.max_resident_pages,
                resident=tuple(
                    decremented if e.page_id == page_id else e
                    for e in self.resident
                ),
            )

        return ResidencyManager(
            eviction_policy=self.eviction_policy,
            max_resident_pages=self.max_resident_pages,
            resident=tuple(e for e in self.resident if e.page_id != page_id),
        )

    def pin(self, page_id: PageId) -> ResidencyManager:
        """Increment the pin count for a resident page."""
        entry = self.get_entry(page_id)
        if entry is None:
            return self

        pinned = ResidentPageEntry(
            page_id=entry.page_id,
            loaded_at=entry.loaded_at,
            last_accessed=entry.last_accessed,
            pin_count=entry.pin_count + 1,
        )
        return ResidencyManager(
            eviction_policy=self.eviction_policy,
            max_resident_pages=self.max_resident_pages,
            resident=tuple(
                pinned if e.page_id == page_id else e for e in self.resident
            ),
        )

    def touch(self, page_id: PageId) -> ResidencyManager:
        """Update the last_accessed timestamp for a resident page."""
        return self._touch(page_id)

    def evict(self) -> tuple[ResidencyManager, PageId | None]:
        """Evict one page if over budget.

        Returns a new ResidencyManager with the victim removed, and the
        evicted page's ID (or None if nothing was evicted).
        """
        victim = self.eviction_policy.select_victim(self.resident)
        if victim is None:
            return self, None

        return (
            ResidencyManager(
                eviction_policy=self.eviction_policy,
                max_resident_pages=self.max_resident_pages,
                resident=tuple(
                    e for e in self.resident if e.page_id != victim.page_id
                ),
            ),
            victim.page_id,
        )

    def evict_all(self) -> ResidencyManager:
        """Evict all unpinned pages."""
        manager = self
        while manager.resident_count() > 0:
            new_manager, victim = manager.evict()
            if victim is None:
                break  # no more evictable pages
            manager = new_manager
        return manager

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _touch(self, page_id: PageId) -> ResidencyManager:
        """Return a new manager with the page's last_accessed updated."""
        now = utc_now()
        return ResidencyManager(
            eviction_policy=self.eviction_policy,
            max_resident_pages=self.max_resident_pages,
            resident=tuple(
                ResidentPageEntry(
                    page_id=e.page_id,
                    loaded_at=e.loaded_at,
                    last_accessed=now if e.page_id == page_id else e.last_accessed,
                    pin_count=e.pin_count,
                )
                for e in self.resident
            ),
        )
