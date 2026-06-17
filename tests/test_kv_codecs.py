"""Tests for rfsn_kv.codecs (identity, quantize, registry)."""

from __future__ import annotations

import pytest

from rfsn_kv.codecs import CODEC_REGISTRY, get_codec
from rfsn_kv.codecs.identity import IdentityCodec
from rfsn_kv.codecs.quantize import QuantizeCodec
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


# ---------------------------------------------------------------------------
# Identity Codec
# ---------------------------------------------------------------------------


class TestIdentityCodec:
    def test_compress_returns_same_data(self) -> None:
        codec = IdentityCodec()
        page = _make_page(b"hello world")
        compressed = codec.compress(page)
        assert compressed.data == b"hello world"
        assert compressed.codec_id == "identity"
        assert compressed.status == "decompressed"

    def test_decompress_is_noop(self) -> None:
        codec = IdentityCodec()
        page = _make_page(b"test data")
        decompressed = codec.decompress(page)
        assert decompressed.data == page.data

    def test_estimate_ratio_is_one(self) -> None:
        codec = IdentityCodec()
        page = _make_page(b"some data here")
        assert codec.estimate_ratio(page) == 1.0

    def test_empty_data(self) -> None:
        codec = IdentityCodec()
        page = _make_page(b"")
        compressed = codec.compress(page)
        assert compressed.data == b""

    def test_preserves_metadata(self) -> None:
        codec = IdentityCodec()
        page = _make_page(b"x")
        compressed = codec.compress(page)
        assert compressed.page_id == page.page_id
        assert compressed.token_offset == page.token_offset
        assert compressed.token_count == page.token_count
        assert compressed.layer_index == page.layer_index


# ---------------------------------------------------------------------------
# Quantize Codec
# ---------------------------------------------------------------------------


class TestQuantizeCodec:
    def test_roundtrip_8bit(self) -> None:
        codec = QuantizeCodec(bit_width=8, group_size=16)
        data = bytes(range(256))[:128]
        page = _make_page(data)
        compressed = codec.compress(page)
        assert compressed.codec_id == "quantize"
        assert compressed.status == "compressed"
        decompressed = codec.decompress(compressed)
        # Quantization introduces small rounding errors; check within tolerance.
        assert len(decompressed.data) == len(data)
        for orig, restored in zip(data, decompressed.data):
            assert abs(int(orig) - int(restored)) <= 1

    def test_roundtrip_4bit(self) -> None:
        codec = QuantizeCodec(bit_width=4, group_size=8)
        data = bytes(range(0, 64))
        page = _make_page(data)
        compressed = codec.compress(page)
        decompressed = codec.decompress(compressed)
        assert len(decompressed.data) == len(data)
        # 4-bit has more rounding error
        for orig, restored in zip(data, decompressed.data):
            assert abs(int(orig) - int(restored)) <= 2

    def test_empty_data(self) -> None:
        codec = QuantizeCodec(bit_width=8, group_size=16)
        page = _make_page(b"")
        compressed = codec.compress(page)
        assert compressed.data == b""
        decompressed = codec.decompress(compressed)
        assert decompressed.data == b""

    def test_single_byte(self) -> None:
        codec = QuantizeCodec(bit_width=8, group_size=64)
        page = _make_page(b"\x42")
        compressed = codec.compress(page)
        decompressed = codec.decompress(compressed)
        assert decompressed.data == b"\x42"

    def test_uniform_data(self) -> None:
        """All-same bytes should roundtrip exactly."""
        codec = QuantizeCodec(bit_width=8, group_size=16)
        data = b"\xff" * 32
        page = _make_page(data)
        compressed = codec.compress(page)
        decompressed = codec.decompress(compressed)
        assert decompressed.data == data

    def test_estimate_ratio(self) -> None:
        codec = QuantizeCodec(bit_width=8, group_size=64)
        # Large data should have ratio close to 1.0 (header overhead amortized)
        page = _make_page(b"\x00" * 1024)
        ratio = codec.estimate_ratio(page)
        assert 0.0 < ratio < 1.5  # may exceed 1.0 for small data due to header

    def test_estimate_ratio_small_data_can_exceed_one(self) -> None:
        """Small data can have ratio > 1.0 due to header overhead."""
        codec = QuantizeCodec(bit_width=8, group_size=64)
        page = _make_page(b"\x00" * 16)
        ratio = codec.estimate_ratio(page)
        assert ratio > 0.0

    def test_invalid_bit_width_rejected(self) -> None:
        with pytest.raises(ValueError, match="bit_width must be 4 or 8"):
            QuantizeCodec(bit_width=2)

    def test_invalid_group_size_rejected(self) -> None:
        with pytest.raises(ValueError, match="group_size must be >= 2"):
            QuantizeCodec(group_size=1)

    def test_preserves_metadata(self) -> None:
        codec = QuantizeCodec(bit_width=8, group_size=16)
        page = _make_page(b"\x00" * 16)
        compressed = codec.compress(page)
        assert compressed.page_id == page.page_id
        assert compressed.token_offset == page.token_offset
        assert compressed.token_count == page.token_count
        assert compressed.layer_index == page.layer_index

    def test_compressed_is_smaller_for_varying_data(self) -> None:
        """For 4-bit quantization, compressed size should be smaller for large data."""
        codec = QuantizeCodec(bit_width=4, group_size=64)
        data = bytes(range(256)) * 4  # 1024 bytes
        page = _make_page(data)
        compressed = codec.compress(page)
        # 4-bit should be roughly half the size
        assert len(compressed.data) < len(data)


# ---------------------------------------------------------------------------
# Codec Registry
# ---------------------------------------------------------------------------


class TestCodecRegistry:
    def test_get_identity(self) -> None:
        codec = get_codec("identity")
        assert isinstance(codec, IdentityCodec)

    def test_get_quantize(self) -> None:
        codec = get_codec("quantize")
        assert isinstance(codec, QuantizeCodec)

    def test_unknown_codec_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown codec"):
            get_codec("nonexistent")

    def test_registry_has_expected_keys(self) -> None:
        assert "identity" in CODEC_REGISTRY
        assert "quantize" in CODEC_REGISTRY
