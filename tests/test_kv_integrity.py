"""Tests for rfsn_kv.integrity."""

from __future__ import annotations

from rfsn_kv.common import hash_bytes
from rfsn_kv.integrity import IntegrityChecker, IntegrityReport
from rfsn_kv.pages import KVPage
from rfsn_kv.types import LayerIndex, PageId


def _make_page(data: bytes) -> KVPage:
    return KVPage.create(
        page_id=PageId("test-page"),
        data=data,
        token_offset=0,
        token_count=10,
        layer_index=LayerIndex(0),
    )


def _make_tampered_page(data: bytes, bad_hash: str, page_id: str = "test-page") -> KVPage:
    """Create a page and then tamper with data so the hash is stale."""
    page = _make_page(b"placeholder")
    object.__setattr__(page, "page_id", PageId(page_id))
    object.__setattr__(page, "data", data)
    object.__setattr__(page, "data_hash", bad_hash)
    return page


class TestIntegrityChecker:
    def test_verify_valid_page(self) -> None:
        checker = IntegrityChecker()
        page = _make_page(b"valid data")
        assert checker.verify_page(page) is True

    def test_verify_corrupt_page(self) -> None:
        checker = IntegrityChecker()
        # Create a valid page, then tamper its data leaving hash stale.
        bad_hash = "0" * 64
        corrupted = _make_tampered_page(b"tampered data", bad_hash)
        assert checker.verify_page(corrupted) is False

    def test_verify_empty_page(self) -> None:
        checker = IntegrityChecker()
        page = _make_page(b"")
        assert checker.verify_page(page) is True

    def test_verify_range_all_valid(self) -> None:
        checker = IntegrityChecker()
        pages = (_make_page(b"a"), _make_page(b"b"), _make_page(b"c"))
        report = checker.verify_page_range(pages)
        assert report.checked == 3
        assert report.valid == 3
        assert report.corrupted == ()

    def test_verify_range_some_corrupt(self) -> None:
        checker = IntegrityChecker()
        good = _make_page(b"good")
        bad = _make_tampered_page(b"bad", "0" * 64, page_id="bad")
        report = checker.verify_page_range((good, bad))
        assert report.checked == 2
        assert report.valid == 1
        assert report.corrupted == (PageId("bad"),)

    def test_verify_range_empty(self) -> None:
        checker = IntegrityChecker()
        report = checker.verify_page_range(())
        assert report.checked == 0
        assert report.valid == 0
        assert report.corrupted == ()

    def test_repair_page_fixes_hash(self) -> None:
        checker = IntegrityChecker()
        bad = _make_tampered_page(b"fix this", "0" * 64)
        repaired = checker.repair_page(bad)
        assert repaired.data_hash == hash_bytes(b"fix this")
        assert repaired.data == b"fix this"

    def test_repair_already_valid_returns_same(self) -> None:
        checker = IntegrityChecker()
        page = _make_page(b"already good")
        repaired = checker.repair_page(page)
        assert repaired is page  # same object if no repair needed

    def test_repair_preserves_metadata(self) -> None:
        checker = IntegrityChecker()
        bad = _make_tampered_page(b"data", "0" * 64)
        # Set metadata fields directly
        object.__setattr__(bad, "token_offset", 42)
        object.__setattr__(bad, "token_count", 7)
        object.__setattr__(bad, "layer_index", LayerIndex(3))
        repaired = checker.repair_page(bad)
        assert repaired.page_id == PageId("test-page")
        assert repaired.token_offset == 42
        assert repaired.token_count == 7
        assert repaired.layer_index == LayerIndex(3)

    def test_integrity_report_dataclass(self) -> None:
        report = IntegrityReport(
            checked=10,
            corrupted=(PageId("a"), PageId("b")),
            valid=8,
        )
        assert report.checked == 10
        assert len(report.corrupted) == 2
        assert report.valid == 8
