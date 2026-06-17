"""Tests for rfsn_kv.persistence."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from rfsn_kv.pages import KVPage
from rfsn_kv.persistence import IntegrityError, KVPersistence
from rfsn_kv.types import LayerIndex, PageId


def _make_page(
    page_id: str,
    data: bytes,
    token_offset: int = 0,
    token_count: int = 10,
    layer_index: int = 0,
) -> KVPage:
    return KVPage.create(
        page_id=PageId(page_id),
        data=data,
        token_offset=token_offset,
        token_count=token_count,
        layer_index=LayerIndex(layer_index),
    )


class TestKVPersistence:
    def test_put_and_get(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "test.db"
            with KVPersistence(db) as store:
                page = _make_page("p-0", b"hello world")
                store.put_page(page)
                retrieved = store.get_page(PageId("p-0"))
                assert retrieved.data == b"hello world"
                assert retrieved.token_count == 10

    def test_get_missing_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "test.db"
            with KVPersistence(db) as store:
                with pytest.raises(KeyError, match="not found"):
                    store.get_page(PageId("missing"))

    def test_has_page(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "test.db"
            with KVPersistence(db) as store:
                assert not store.has_page(PageId("p-0"))
                store.put_page(_make_page("p-0", b"data"))
                assert store.has_page(PageId("p-0"))

    def test_put_idempotent_same_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "test.db"
            with KVPersistence(db) as store:
                page = _make_page("p-0", b"data")
                store.put_page(page)
                # Should not raise
                store.put_page(page)
                assert store.count_pages() == 1

    def test_put_conflicting_hash_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "test.db"
            with KVPersistence(db) as store:
                page1 = _make_page("p-0", b"original")
                store.put_page(page1)
                page2 = _make_page("p-0", b"different")
                with pytest.raises(IntegrityError):
                    store.put_page(page2)

    def test_delete_page(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "test.db"
            with KVPersistence(db) as store:
                store.put_page(_make_page("p-0", b"data"))
                assert store.delete_page(PageId("p-0"))
                assert not store.has_page(PageId("p-0"))

    def test_delete_nonexistent_returns_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "test.db"
            with KVPersistence(db) as store:
                assert not store.delete_page(PageId("missing"))

    def test_list_pages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "test.db"
            with KVPersistence(db) as store:
                store.put_page(_make_page("p-2", b"d"))
                store.put_page(_make_page("p-0", b"a"))
                store.put_page(_make_page("p-1", b"c"))
                pages = store.list_pages()
                assert pages == (PageId("p-0"), PageId("p-1"), PageId("p-2"))

    def test_list_pages_for_layer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "test.db"
            with KVPersistence(db) as store:
                store.put_page(_make_page("p-0", b"a", layer_index=0))
                store.put_page(_make_page("p-1", b"b", layer_index=1))
                store.put_page(_make_page("p-2", b"c", layer_index=0))
                layer0 = store.list_pages_for_layer(LayerIndex(0))
                assert layer0 == (PageId("p-0"), PageId("p-2"))

    def test_count_pages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "test.db"
            with KVPersistence(db) as store:
                assert store.count_pages() == 0
                store.put_page(_make_page("p-0", b"a"))
                store.put_page(_make_page("p-1", b"b"))
                assert store.count_pages() == 2

    def test_preserves_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "test.db"
            with KVPersistence(db) as store:
                page = KVPage.create(
                    page_id=PageId("p-0"),
                    data=b"\x00" * 16,
                    token_offset=100,
                    token_count=16,
                    layer_index=LayerIndex(3),
                    head_range=(0, 5, 10),
                    codec_id="quantize",
                    status="compressed",
                    actor="test-actor",
                    action_id="test-action",
                )
                store.put_page(page)
                retrieved = store.get_page(PageId("p-0"))
                assert retrieved.token_offset == 100
                assert retrieved.layer_index == LayerIndex(3)
                assert retrieved.head_range == (0, 5, 10)
                assert retrieved.codec_id == "quantize"
                assert retrieved.status == "compressed"
                assert retrieved.actor == "test-actor"

    def test_large_page(self) -> None:
        """Pages up to 1MB should round-trip correctly."""
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "test.db"
            with KVPersistence(db) as store:
                data = b"\x00" * (1024 * 1024)
                page = _make_page("large", data, token_count=1000)
                store.put_page(page)
                retrieved = store.get_page(PageId("large"))
                assert retrieved.data == data
