"""Simulated quantization codec for byte-level KV page data.

This codec simulates symmetric min-max quantization at the byte level.
For the MVP, it works on raw bytes rather than actual float tensors.
The actual float-tensor compression (per-layer, per-group min-max with
dequantization) will be implemented when MLX kernels are available in
Phase 10.

The simulation divides the byte data into groups of ``group_size`` bytes
and stores each group as ``bit_width``-bit indices relative to the group
min/max. The compressed output includes a header with group metadata
followed by packed indices.
"""

from __future__ import annotations

import struct

from rfsn_kv.common import hash_bytes
from rfsn_kv.pages import KVPage

_HEADER_FMT = ">II"  # group_size (uint32), num_groups (uint32)
_GROUP_HEADER_FMT = ">BB"  # min_byte (uint8), max_byte (uint8)


class QuantizeCodec:
    """Byte-level quantization codec with configurable bit width and group size.

    Attributes:
        bit_width: Number of bits per element (4 or 8).
        group_size: Number of bytes per quantization group.
    """

    codec_id: str = "quantize"
    codec_version: int = 1

    def __init__(self, bit_width: int = 8, group_size: int = 64) -> None:
        if bit_width not in (4, 8):
            raise ValueError(f"bit_width must be 4 or 8, got {bit_width}")
        if group_size < 2:
            raise ValueError(f"group_size must be >= 2, got {group_size}")
        self.bit_width = bit_width
        self.group_size = group_size

    def compress(self, page: KVPage) -> KVPage:
        """Quantize page data using symmetric min-max quantization."""
        data = page.data
        if len(data) == 0:
            return KVPage(
                page_id=page.page_id,
                data=b"",
                data_hash=hash_bytes(b""),
                token_offset=page.token_offset,
                token_count=page.token_count,
                layer_index=page.layer_index,
                head_range=page.head_range,
                codec_id="quantize",
                status="compressed",
                created_at=page.created_at,
                actor=page.actor,
                action_id=page.action_id,
                event_id=page.event_id,
            )

        groups = self._split_groups(data)
        out_parts: list[bytes] = [
            struct.pack(_HEADER_FMT, self.group_size, len(groups))
        ]

        max_index = (1 << self.bit_width) - 1
        for group in groups:
            g_min = min(group)
            g_max = max(group)
            out_parts.append(struct.pack(_GROUP_HEADER_FMT, g_min, g_max))
            span = g_max - g_min
            if span == 0:
                indices = [0] * len(group)
            else:
                indices = [
                    round((b - g_min) / span * max_index) for b in group
                ]
            if self.bit_width == 4:
                packed = self._pack_4bit(indices)
                out_parts.append(packed)
            else:
                out_parts.append(bytes(indices))

        compressed = b"".join(out_parts)
        return KVPage(
            page_id=page.page_id,
            data=compressed,
            data_hash=hash_bytes(compressed),
            token_offset=page.token_offset,
            token_count=page.token_count,
            layer_index=page.layer_index,
            head_range=page.head_range,
            codec_id="quantize",
            status="compressed",
            created_at=page.created_at,
            actor=page.actor,
            action_id=page.action_id,
            event_id=page.event_id,
        )

    def decompress(self, page: KVPage) -> KVPage:
        """Dequantize page data back to the original byte representation."""
        data = page.data
        if len(data) == 0:
            return KVPage(
                page_id=page.page_id,
                data=b"",
                data_hash=hash_bytes(b""),
                token_offset=page.token_offset,
                token_count=page.token_count,
                layer_index=page.layer_index,
                head_range=page.head_range,
                codec_id="identity",
                status="decompressed",
                created_at=page.created_at,
                actor=page.actor,
                action_id=page.action_id,
                event_id=page.event_id,
            )

        group_size_stored, num_groups = struct.unpack(_HEADER_FMT, data[:8])
        offset = 8
        max_index = (1 << self.bit_width) - 1
        out_parts: list[bytes] = []

        for _ in range(num_groups):
            g_min, g_max = struct.unpack(
                _GROUP_HEADER_FMT, data[offset : offset + 2]
            )
            offset += 2
            group_len = group_size_stored  # all groups are full except possibly the last
            if self.bit_width == 4:
                num_indices = group_len
                packed_len = (num_indices + 1) // 2
                indices = self._unpack_4bit(data[offset : offset + packed_len], num_indices)
                offset += packed_len
            else:
                indices = list(data[offset : offset + group_len])
                offset += group_len

            span = g_max - g_min
            restored = bytes(
                g_min + round(idx / max_index * span) if max_index > 0 else g_min
                for idx in indices
            )
            out_parts.append(restored)

        decompressed = b"".join(out_parts)
        return KVPage(
            page_id=page.page_id,
            data=decompressed,
            data_hash=hash_bytes(decompressed),
            token_offset=page.token_offset,
            token_count=page.token_count,
            layer_index=page.layer_index,
            head_range=page.head_range,
            codec_id="identity",
            status="decompressed",
            created_at=page.created_at,
            actor=page.actor,
            action_id=page.action_id,
            event_id=page.event_id,
        )

    def estimate_ratio(self, page: KVPage) -> float:
        """Estimate the compression ratio."""
        if len(page.data) == 0:
            return 1.0
        num_groups = (len(page.data) + self.group_size - 1) // self.group_size
        header = 8  # uint32 * 2
        per_group = 2 + (self.group_size // 2 if self.bit_width == 4 else self.group_size)
        compressed = header + num_groups * per_group
        return compressed / len(page.data)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_groups(self, data: bytes) -> list[tuple[int, ...]]:
        return [
            tuple(data[i : i + self.group_size])
            for i in range(0, len(data), self.group_size)
        ]

    @staticmethod
    def _pack_4bit(indices: list[int]) -> bytes:
        """Pack 4-bit indices into bytes (two per byte, high nibble first)."""
        packed = bytearray()
        for i in range(0, len(indices), 2):
            high = indices[i] & 0x0F
            low = (indices[i + 1] & 0x0F) if i + 1 < len(indices) else 0
            packed.append((high << 4) | low)
        return bytes(packed)

    @staticmethod
    def _unpack_4bit(data: bytes, num_indices: int) -> list[int]:
        """Unpack 4-bit indices from bytes."""
        indices: list[int] = []
        for byte in data:
            indices.append((byte >> 4) & 0x0F)
            if len(indices) < num_indices:
                indices.append(byte & 0x0F)
        return indices[:num_indices]
