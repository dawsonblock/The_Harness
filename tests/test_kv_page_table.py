"""Tests for rfsn_kv.page_table."""

from __future__ import annotations

import pytest

from rfsn_kv.page_table import PageTable, PageTableEntry
from rfsn_kv.types import LayerIndex, PageId


def _entry(
    logical_position: int,
    page_id: str,
    token_offset: int = 0,
    token_count: int = 10,
    layer_index: int = 0,
) -> PageTableEntry:
    return PageTableEntry(
        logical_position=logical_position,
        page_id=PageId(page_id),
        token_offset=token_offset,
        token_count=token_count,
        layer_index=LayerIndex(layer_index),
    )


class TestPageTableEntry:
    def test_create_and_valid(self) -> None:
        e = _entry(0, "p-0")
        assert e.logical_position == 0
        assert e.page_id == "p-0"
        assert e.logical_end == 10

    def test_negative_logical_position_rejected(self) -> None:
        with pytest.raises(ValueError, match="logical_position must be >= 0"):
            PageTableEntry(
                logical_position=-1,
                page_id=PageId("p-0"),
                token_offset=0,
                token_count=5,
                layer_index=LayerIndex(0),
            )

    def test_negative_token_offset_rejected(self) -> None:
        with pytest.raises(ValueError, match="token_offset must be >= 0"):
            PageTableEntry(
                logical_position=0,
                page_id=PageId("p-0"),
                token_offset=-1,
                token_count=5,
                layer_index=LayerIndex(0),
            )

    def test_zero_token_count_rejected(self) -> None:
        with pytest.raises(ValueError, match="token_count must be > 0"):
            PageTableEntry(
                logical_position=0,
                page_id=PageId("p-0"),
                token_offset=0,
                token_count=0,
                layer_index=LayerIndex(0),
            )

    def test_logical_end(self) -> None:
        e = _entry(10, "p-0", token_count=20)
        assert e.logical_end == 30


class TestPageTable:
    def test_empty_table(self) -> None:
        pt = PageTable()
        assert len(pt) == 0
        assert pt.entries == ()

    def test_single_entry(self) -> None:
        pt = PageTable(entries=(_entry(0, "p-0"),))
        assert len(pt) == 1

    def test_sorted_entries_accepted(self) -> None:
        pt = PageTable(entries=(_entry(0, "p-0"), _entry(10, "p-1"), _entry(20, "p-2")))
        assert len(pt) == 3

    def test_unsorted_entries_rejected(self) -> None:
        with pytest.raises(ValueError, match="sorted by logical_position"):
            PageTable(entries=(_entry(10, "p-1"), _entry(0, "p-0")))

    def test_duplicate_positions_rejected(self) -> None:
        with pytest.raises(ValueError, match="sorted by logical_position"):
            PageTable(entries=(_entry(0, "p-0"), _entry(0, "p-1")))

    def test_lookup_exact_match(self) -> None:
        pt = PageTable(entries=(_entry(0, "p-0"), _entry(10, "p-1")))
        result = pt.lookup(0)
        assert result is not None
        assert result.page_id == "p-0"

    def test_lookup_within_range(self) -> None:
        pt = PageTable(entries=(_entry(0, "p-0", token_count=20),))
        assert pt.lookup(5) is not None
        assert pt.lookup(19) is not None

    def test_lookup_not_found(self) -> None:
        pt = PageTable(entries=(_entry(0, "p-0", token_count=10),))
        assert pt.lookup(10) is None
        assert pt.lookup(-1) is None

    def test_lookup_between_entries(self) -> None:
        pt = PageTable(
            entries=(
                _entry(0, "p-0", token_count=5),
                _entry(10, "p-1", token_count=5),
            )
        )
        assert pt.lookup(7) is None

    def test_range_query_single_page(self) -> None:
        pt = PageTable(entries=(_entry(0, "p-0", token_count=100),))
        pr = pt.range_query(0, 50)
        assert pr.page_ids == (PageId("p-0"),)

    def test_range_query_multiple_pages(self) -> None:
        pt = PageTable(
            entries=(
                _entry(0, "p-0", token_count=10),
                _entry(10, "p-1", token_count=10),
                _entry(20, "p-2", token_count=10),
            )
        )
        pr = pt.range_query(5, 25)
        assert pr.page_ids == (PageId("p-0"), PageId("p-1"), PageId("p-2"))

    def test_range_query_rejected_when_end_not_greater(self) -> None:
        pt = PageTable(entries=(_entry(0, "p-0"),))
        with pytest.raises(ValueError, match="end_token.*must be >"):
            pt.range_query(10, 10)

    def test_with_entry_adds_in_sorted_order(self) -> None:
        pt = PageTable(entries=(_entry(0, "p-0"), _entry(20, "p-2")))
        new_entry = _entry(10, "p-1")
        pt2 = pt.with_entry(new_entry)
        assert len(pt2) == 3
        assert pt2.entries[1].page_id == "p-1"

    def test_with_entry_preserves_immutability(self) -> None:
        pt = PageTable(entries=(_entry(0, "p-0"),))
        pt2 = pt.with_entry(_entry(10, "p-1"))
        assert len(pt) == 1
        assert len(pt2) == 2

    def test_without_entries_removes_matching(self) -> None:
        pt = PageTable(
            entries=(_entry(0, "p-0"), _entry(10, "p-1"), _entry(20, "p-2"))
        )
        pt2 = pt.without_entries(frozenset({PageId("p-1")}))
        assert len(pt2) == 2
        assert all(e.page_id != "p-1" for e in pt2.entries)

    def test_without_entries_no_match(self) -> None:
        pt = PageTable(entries=(_entry(0, "p-0"),))
        pt2 = pt.without_entries(frozenset({PageId("p-99")}))
        assert len(pt2) == 1

    def test_without_entries_preserves_immutability(self) -> None:
        pt = PageTable(
            entries=(_entry(0, "p-0"), _entry(10, "p-1"))
        )
        pt2 = pt.without_entries(frozenset({PageId("p-0")}))
        assert len(pt) == 2
        assert len(pt2) == 1

    def test_iter(self) -> None:
        entries = (_entry(0, "p-0"), _entry(10, "p-1"))
        pt = PageTable(entries=entries)
        assert list(pt) == list(entries)

    def test_table_hash_deterministic(self) -> None:
        entries = (_entry(0, "p-0"), _entry(10, "p-1"))
        a = PageTable(entries=entries)
        b = PageTable(entries=entries)
        assert a.table_hash == b.table_hash

    def test_table_hash_changes_with_entries(self) -> None:
        a = PageTable(entries=(_entry(0, "p-0"),))
        b = PageTable(entries=(_entry(0, "p-0"), _entry(10, "p-1")))
        assert a.table_hash != b.table_hash
