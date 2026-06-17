"""Tests for rfsn_kv.types, rfsn_kv.common, and rfsn_kv.pages."""

from __future__ import annotations

import pytest

from rfsn_kv.common import (
    canonical_json,
    dataclass_from_dict,
    dataclass_to_dict,
    hash_bytes,
    hash_content,
    sha256_hash,
)
from rfsn_kv.pages import KVPage, PageRange
from rfsn_kv.types import ContentHash, EvictionState, KVPageStatus, LayerIndex, PageId

# ---------------------------------------------------------------------------
# types.py
# ---------------------------------------------------------------------------


class TestTypes:
    def test_page_id_is_string(self) -> None:
        pid = PageId("page-001")
        assert pid == "page-001"

    def test_layer_index_is_int(self) -> None:
        idx = LayerIndex(3)
        assert idx == 3

    def test_content_hash_is_string(self) -> None:
        h = ContentHash("abc123")
        assert h == "abc123"

    def test_kv_page_status_enum(self) -> None:
        assert KVPageStatus.COMPRESSED == "compressed"
        assert KVPageStatus.DECOMPRESSED == "decompressed"
        assert KVPageStatus.EVICTED == "evicted"

    def test_eviction_state_enum(self) -> None:
        assert EvictionState.PINNED == "pinned"
        assert EvictionState.RESIDENT == "resident"
        assert EvictionState.EVICTED == "evicted"


# ---------------------------------------------------------------------------
# common.py
# ---------------------------------------------------------------------------


class TestCommon:
    def test_sha256_deterministic(self) -> None:
        a = sha256_hash("hello")
        b = sha256_hash("hello")
        assert a == b
        assert len(a) == 64

    def test_sha256_bytes(self) -> None:
        h = sha256_hash(b"hello")
        assert h == sha256_hash("hello")

    def test_hash_content_returns_content_hash(self) -> None:
        h = hash_content("test")
        # NewType is a type alias at runtime; value is a str
        assert isinstance(h, str)
        assert h == sha256_hash("test")

    def test_hash_bytes_returns_content_hash(self) -> None:
        h = hash_bytes(b"test")
        assert isinstance(h, str)
        assert h == sha256_hash(b"test")

    def test_canonical_json_sorted_keys(self) -> None:
        result = canonical_json({"b": 2, "a": 1})
        assert result == '{"a":1,"b":2}'

    def test_canonical_json_deterministic(self) -> None:
        a = canonical_json({"x": [1, 2, 3]})
        b = canonical_json({"x": [1, 2, 3]})
        assert a == b

    def test_dataclass_round_trip(self) -> None:
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class Simple:
            name: str
            value: int

        obj = Simple(name="test", value=42)
        d = dataclass_to_dict(obj)
        restored = dataclass_from_dict(Simple, d)
        assert restored == obj

    def test_canonical_json_tuples(self) -> None:
        result = canonical_json({"items": [1, 2, 3]})
        assert result == '{"items":[1,2,3]}'


# ---------------------------------------------------------------------------
# pages.py
# ---------------------------------------------------------------------------


class TestKVPage:
    def test_create_and_immutable(self) -> None:
        page = KVPage.create(
            page_id=PageId("p-0"),
            data=b"hello world",
            token_offset=0,
            token_count=10,
            layer_index=LayerIndex(0),
        )
        assert page.page_id == "p-0"
        assert page.data == b"hello world"
        assert page.token_count == 10
        assert page.compressed_size == 11
        assert page.token_end == 10
        assert page.codec_id == "identity"
        with pytest.raises(AttributeError):
            page.data = b"mutated"  # type: ignore[misc]

    def test_data_hash_validated(self) -> None:
        bad_hash = ContentHash("0" * 64)
        with pytest.raises(ValueError, match="data_hash mismatch"):
            KVPage(
                page_id=PageId("p-1"),
                data=b"hello",
                data_hash=bad_hash,
                token_offset=0,
                token_count=5,
                layer_index=LayerIndex(0),
            )

    def test_negative_token_offset_rejected(self) -> None:
        with pytest.raises(ValueError, match="token_offset must be >= 0"):
            KVPage.create(
                page_id=PageId("p-2"),
                data=b"x",
                token_offset=-1,
                token_count=1,
                layer_index=LayerIndex(0),
            )

    def test_zero_token_count_rejected(self) -> None:
        with pytest.raises(ValueError, match="token_count must be > 0"):
            KVPage.create(
                page_id=PageId("p-3"),
                data=b"x",
                token_offset=0,
                token_count=0,
                layer_index=LayerIndex(0),
            )

    def test_head_range_default_empty(self) -> None:
        page = KVPage.create(
            page_id=PageId("p-4"),
            data=b"x",
            token_offset=0,
            token_count=1,
            layer_index=LayerIndex(0),
        )
        assert page.head_range == ()

    def test_head_range_specified(self) -> None:
        page = KVPage.create(
            page_id=PageId("p-5"),
            data=b"x",
            token_offset=0,
            token_count=1,
            layer_index=LayerIndex(2),
            head_range=(0, 3, 7),
        )
        assert page.head_range == (0, 3, 7)

    def test_empty_data_valid(self) -> None:
        page = KVPage.create(
            page_id=PageId("p-6"),
            data=b"",
            token_offset=0,
            token_count=1,
            layer_index=LayerIndex(0),
        )
        assert page.data == b""
        assert page.compressed_size == 0

    def test_default_provenance(self) -> None:
        page = KVPage.create(
            page_id=PageId("p-7"),
            data=b"test",
            token_offset=0,
            token_count=1,
            layer_index=LayerIndex(0),
        )
        assert page.actor == "system"
        assert page.action_id == "init"
        assert page.event_id is None


class TestPageRange:
    def test_create_and_valid(self) -> None:
        pr = PageRange(start_token=0, end_token=100, page_ids=(PageId("a"), PageId("b")))
        assert pr.start_token == 0
        assert pr.end_token == 100
        assert len(pr.page_ids) == 2

    def test_empty_page_ids(self) -> None:
        pr = PageRange(start_token=0, end_token=50)
        assert pr.page_ids == ()

    def test_rejected_when_end_not_greater(self) -> None:
        with pytest.raises(ValueError, match="end_token.*must be >"):
            PageRange(start_token=10, end_token=10)

    def test_rejected_when_end_less_than_start(self) -> None:
        with pytest.raises(ValueError, match="end_token.*must be >"):
            PageRange(start_token=10, end_token=5)


# ---------------------------------------------------------------------------
# CAS offloading
# ---------------------------------------------------------------------------

class _FakeCAS:
    """Minimal in-memory CAS store for testing."""

    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}

    def put(self, data: bytes | str) -> str:
        if isinstance(data, str):
            data = data.encode("utf-8")
        from rfsn_kv.common import sha256_hash
        h = sha256_hash(data)
        self._store[h] = data
        return h

    def put_text(self, text: str) -> str:
        return self.put(text.encode("utf-8"))

    def get_text(self, content_hash: str) -> str:
        return self._store[content_hash].decode("utf-8")

    def get_bytes(self, content_hash: str) -> bytes:
        return self._store[content_hash]


class TestKVPageCAS:
    def test_create_with_cas_offloads_data(self) -> None:
        cas = _FakeCAS()
        page = KVPage.create_with_cas(
            page_id=PageId("p-0"),
            data=b"hello world",
            token_offset=0,
            token_count=10,
            layer_index=LayerIndex(0),
            cas=cas,
        )
        assert page.data == b""
        assert page.is_offloaded
        assert page.content_ref is not None
        assert page.content_ref.byte_length == 11

    def test_resolve_content_retrieves_from_cas(self) -> None:
        cas = _FakeCAS()
        page = KVPage.create_with_cas(
            page_id=PageId("p-0"),
            data=b"hello world",
            token_offset=0,
            token_count=10,
            layer_index=LayerIndex(0),
            cas=cas,
        )
        resolved = page.resolve_content(cas)
        assert resolved == b"hello world"

    def test_inline_page_resolve_returns_data(self) -> None:
        cas = _FakeCAS()
        page = KVPage.create(
            page_id=PageId("p-0"),
            data=b"inline data",
            token_offset=0,
            token_count=10,
            layer_index=LayerIndex(0),
        )
        assert not page.is_offloaded
        assert page.resolve_content(cas) == b"inline data"

    def test_create_with_cas_preserves_metadata(self) -> None:
        cas = _FakeCAS()
        page = KVPage.create_with_cas(
            page_id=PageId("p-0"),
            data=b"data",
            token_offset=42,
            token_count=7,
            layer_index=LayerIndex(3),
            head_range=(0, 5),
            codec_id="quantize",
            status="compressed",
            actor="test-actor",
            action_id="test-action",
            cas=cas,
        )
        assert page.token_offset == 42
        assert page.token_count == 7
        assert page.layer_index == LayerIndex(3)
        assert page.head_range == (0, 5)
        assert page.codec_id == "quantize"
        assert page.status == "compressed"
        assert page.actor == "test-actor"
        assert page.action_id == "test-action"

    def test_data_hash_matches_cas_content(self) -> None:
        from rfsn_kv.common import hash_bytes
        cas = _FakeCAS()
        data = b"test content"
        page = KVPage.create_with_cas(
            page_id=PageId("p-0"),
            data=data,
            token_offset=0,
            token_count=5,
            layer_index=LayerIndex(0),
            cas=cas,
        )
        assert page.data_hash == hash_bytes(data)
