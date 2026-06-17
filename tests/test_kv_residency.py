"""Tests for rfsn_kv.residency."""

from __future__ import annotations

import pytest

from rfsn_kv.residency import (
    LRUEvictionPolicy,
    ResidencyManager,
    ResidentPageEntry,
)
from rfsn_kv.types import PageId


class TestResidentPageEntry:
    def test_create(self) -> None:
        entry = ResidentPageEntry(page_id=PageId("p-0"))
        assert entry.page_id == "p-0"
        assert entry.pin_count == 0
        assert entry.is_pinned is False

    def test_pinned(self) -> None:
        entry = ResidentPageEntry(page_id=PageId("p-0"), pin_count=1)
        assert entry.is_pinned is True

    def test_immutable(self) -> None:
        entry = ResidentPageEntry(page_id=PageId("p-0"))
        with pytest.raises(AttributeError):
            entry.page_id = PageId("p-1")  # type: ignore[misc]


class TestLRUEvictionPolicy:
    def test_select_victim_lru(self) -> None:
        from datetime import timedelta

        from rfsn_kv.common import utc_now

        now = utc_now()
        a = ResidentPageEntry(
            page_id=PageId("a"),
            last_accessed=now - timedelta(minutes=10),
        )
        b = ResidentPageEntry(
            page_id=PageId("b"),
            last_accessed=now,
        )
        policy = LRUEvictionPolicy()
        victim = policy.select_victim((a, b))
        assert victim is not None
        assert victim.page_id == "a"

    def test_select_victim_skips_pinned(self) -> None:
        from datetime import timedelta

        from rfsn_kv.common import utc_now

        now = utc_now()
        pinned = ResidentPageEntry(
            page_id=PageId("a"),
            last_accessed=now - timedelta(minutes=10),
            pin_count=1,
        )
        unpinned = ResidentPageEntry(
            page_id=PageId("b"),
            last_accessed=now,
        )
        policy = LRUEvictionPolicy()
        victim = policy.select_victim((pinned, unpinned))
        assert victim is not None
        assert victim.page_id == "b"

    def test_select_victim_all_pinned(self) -> None:
        pinned = ResidentPageEntry(page_id=PageId("a"), pin_count=1)
        policy = LRUEvictionPolicy()
        victim = policy.select_victim((pinned,))
        assert victim is None

    def test_select_victim_empty(self) -> None:
        policy = LRUEvictionPolicy()
        victim = policy.select_victim(())
        assert victim is None


class TestResidencyManager:
    def test_empty_manager(self) -> None:
        rm = ResidencyManager()
        assert rm.resident_count() == 0
        assert rm.is_resident(PageId("p-0")) is False

    def test_load_page(self) -> None:
        rm = ResidencyManager(max_resident_pages=10)
        rm2 = rm.load(PageId("p-0"))
        assert rm.resident_count() == 0  # original unchanged
        assert rm2.resident_count() == 1
        assert rm2.is_resident(PageId("p-0"))

    def test_load_existing_touches(self) -> None:
        rm = ResidencyManager(max_resident_pages=10)
        rm2 = rm.load(PageId("p-0"))
        rm3 = rm2.load(PageId("p-0"))
        assert rm3.resident_count() == 1

    def test_unload_page(self) -> None:
        rm = ResidencyManager(max_resident_pages=10)
        rm2 = rm.load(PageId("p-0"))
        rm3 = rm2.unload(PageId("p-0"))
        assert rm3.resident_count() == 0

    def test_unload_nonexistent(self) -> None:
        rm = ResidencyManager(max_resident_pages=10)
        rm2 = rm.load(PageId("p-0"))
        rm3 = rm2.unload(PageId("p-99"))
        assert rm3.resident_count() == 1

    def test_eviction_when_full(self) -> None:
        rm = ResidencyManager(max_resident_pages=2)
        rm2 = rm.load(PageId("p-0"))
        rm3 = rm2.load(PageId("p-1"))
        rm4 = rm3.load(PageId("p-2"))  # triggers eviction
        assert rm4.resident_count() == 2
        assert rm4.is_resident(PageId("p-2"))

    def test_evict_returns_victim(self) -> None:
        from datetime import timedelta

        from rfsn_kv.common import utc_now

        now = utc_now()
        rm = ResidencyManager(max_resident_pages=10)
        rm2 = rm.load(PageId("p-0"))
        # Manually set older timestamp for p-0
        old_entry = ResidentPageEntry(
            page_id=PageId("p-0"),
            loaded_at=now - timedelta(minutes=10),
            last_accessed=now - timedelta(minutes=10),
        )
        rm3 = ResidencyManager(
            eviction_policy=rm2.eviction_policy,
            max_resident_pages=rm2.max_resident_pages,
            resident=(old_entry,),
        )
        rm4 = rm3.load(PageId("p-1"))
        rm5, victim = rm4.evict()
        assert victim == PageId("p-0")
        assert rm5.resident_count() == 1

    def test_pin_prevents_eviction(self) -> None:
        rm = ResidencyManager(max_resident_pages=1)
        rm2 = rm.load(PageId("p-0"))
        rm3 = rm2.pin(PageId("p-0"))
        rm4 = rm3.load(PageId("p-1"))  # should evict nothing since p-0 is pinned
        # p-0 is pinned, so it should still be there
        assert rm4.is_resident(PageId("p-0"))

    def test_pin_unload_decrements_pin(self) -> None:
        rm = ResidencyManager(max_resident_pages=10)
        rm2 = rm.load(PageId("p-0"))
        rm3 = rm2.pin(PageId("p-0"))
        rm4 = rm3.unload(PageId("p-0"))
        # Pin count was 1, unload decrements to 0 but keeps page
        assert rm4.is_resident(PageId("p-0"))
        entry = rm4.get_entry(PageId("p-0"))
        assert entry is not None
        assert entry.pin_count == 0

    def test_touch_updates_timestamp(self) -> None:
        rm = ResidencyManager(max_resident_pages=10)
        rm2 = rm.load(PageId("p-0"))
        entry_before = rm2.get_entry(PageId("p-0"))
        assert entry_before is not None
        rm3 = rm2.touch(PageId("p-0"))
        entry_after = rm3.get_entry(PageId("p-0"))
        assert entry_after is not None
        assert entry_after.last_accessed >= entry_before.last_accessed

    def test_evict_all(self) -> None:
        rm = ResidencyManager(max_resident_pages=10)
        rm2 = rm.load(PageId("p-0"))
        rm3 = rm2.load(PageId("p-1"))
        rm4 = rm3.load(PageId("p-2"))
        rm5 = rm4.pin(PageId("p-0"))
        rm6 = rm5.evict_all()
        # p-0 is pinned, should remain
        assert rm6.is_resident(PageId("p-0"))
        assert not rm6.is_resident(PageId("p-1"))
        assert not rm6.is_resident(PageId("p-2"))

    def test_get_entry(self) -> None:
        rm = ResidencyManager(max_resident_pages=10)
        rm2 = rm.load(PageId("p-0"))
        entry = rm2.get_entry(PageId("p-0"))
        assert entry is not None
        assert entry.page_id == "p-0"
        assert rm2.get_entry(PageId("p-99")) is None

    def test_preserves_immutability(self) -> None:
        rm = ResidencyManager(max_resident_pages=10)
        rm2 = rm.load(PageId("p-0"))
        assert rm.resident_count() == 0
        assert rm2.resident_count() == 1
