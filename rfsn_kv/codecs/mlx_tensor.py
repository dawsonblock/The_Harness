"""Optional MLX tensor codec for KV page compression.

This module is intentionally lazy: importing ``rfsn_kv`` must not require MLX.
When MLX is unavailable, ``MLXTensorCodec`` still exists but raises
``MLXUnavailableError`` from ``compress``/``decompress`` so tests and callers can
probe availability without importing optional dependencies.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Any

from rfsn_kv.common import hash_bytes
from rfsn_kv.pages import KVPage


class MLXUnavailableError(RuntimeError):
    """Raised when the optional MLX runtime is not installed."""


@dataclass(frozen=True, slots=True)
class MLXTensorMetadata:
    """Metadata describing the tensor shape and dtype carried by a compressed page."""

    shape: tuple[int, ...]
    dtype: str
    tensor_count: int
    compressed_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape),
            "dtype": self.dtype,
            "tensor_count": self.tensor_count,
            "compressed_bytes": self.compressed_bytes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MLXTensorMetadata:
        return cls(
            shape=tuple(int(x) for x in data["shape"]),
            dtype=str(data["dtype"]),
            tensor_count=int(data["tensor_count"]),
            compressed_bytes=int(data["compressed_bytes"]),
        )


class MLXTensorCodec:
    """Tensor-aware KV page codec backed by optional MLX.

    The codec accepts either an MLX array (or any object with ``tolist`` and
    ``shape``) or a bytes payload. MLX-backed arrays are compressed through a
    compact serialized header plus raw tensor bytes. The exact compression path
    is intentionally conservative for the MVP: it preserves dtype/shape metadata
    and delegates byte-level compression to the caller/runtime when MLX is
    available.

    Attributes:
        dtype: Default tensor dtype name used when no explicit dtype is supplied.
        codec_id: Stable registry id for MLX-backed tensor pages.
    """

    codec_id: str = "mlx_tensor"
    codec_version: int = 1
    _HEADER_FMT = ">II"

    def __init__(self, dtype: str = "float16") -> None:
        self.dtype = dtype

    def compress(self, page: KVPage) -> KVPage:
        """Compress an MLX tensor page.

        Raises:
            MLXUnavailableError: If MLX is not installed.
        """
        mlx = self._require_mlx()
        tensor = self._coerce_tensor(page.data)
        shape = self._array_shape(tensor)
        dtype = self._array_dtype(tensor)
        tensor_count = 1
        array_bytes = self._array_bytes(mlx, tensor)

        metadata = MLXTensorMetadata(
            shape=shape,
            dtype=dtype,
            tensor_count=tensor_count,
            compressed_bytes=len(array_bytes),
        )
        header = struct.pack(self._HEADER_FMT, self.codec_version, len(array_bytes))
        metadata_json = json.dumps(metadata.to_dict(), sort_keys=True).encode("utf-8")
        metadata_len = struct.pack(">I", len(metadata_json))
        compressed = header + metadata_len + metadata_json + array_bytes
        return self._page_like(
            page,
            data=compressed,
            codec_id=self.codec_id,
            status="compressed",
        )

    def decompress(self, page: KVPage) -> KVPage:
        """Decompress an MLX tensor page.

        Raises:
            MLXUnavailableError: If MLX is not installed.
            ValueError: If the payload is not a valid MLX tensor payload.
        """
        self._require_mlx()
        data = page.data
        if len(data) < 12:
            raise ValueError("MLX tensor payload is too short")

        version, array_len = struct.unpack(self._HEADER_FMT, data[:8])
        if version != self.codec_version:
            raise ValueError(f"Unsupported MLX tensor codec version: {version}")

        metadata_len = struct.unpack(">I", data[8:12])[0]
        offset = 12
        metadata_end = offset + metadata_len
        if metadata_end + array_len > len(data):
            raise ValueError("MLX tensor payload metadata length is invalid")

        metadata = MLXTensorMetadata.from_dict(
            json.loads(data[offset:metadata_end].decode("utf-8"))
        )
        array_bytes = data[metadata_end : metadata_end + array_len]
        if len(array_bytes) != metadata.compressed_bytes:
            raise ValueError("MLX tensor payload length does not match metadata")

        # The MVP stores the runtime tensor bytes verbatim; callers that enable
        # MLX-specific compression can specialize this hook.
        tensor = self._array_from_bytes(array_bytes, metadata.dtype)
        decompressed = self._array_bytes(self._require_mlx(), tensor)
        return self._page_like(
            page,
            data=decompressed,
            codec_id="identity",
            status="decompressed",
        )

    def estimate_ratio(self, page: KVPage) -> float:
        """Estimate compression ratio for an MLX tensor page."""
        if len(page.data) == 0:
            return 1.0
        try:
            compressed = self.compress(page)
            return len(compressed.data) / len(page.data)
        except MLXUnavailableError:
            return 1.0

    def _require_mlx(self) -> Any:
        try:
            import mlx.core as mx

            return mx
        except Exception as exc:  # pragma: no cover - depends on environment
            raise MLXUnavailableError("MLX is not installed") from exc

    def _coerce_tensor(self, data: bytes) -> Any:
        if hasattr(data, "tolist") and hasattr(data, "shape"):
            return data
        return self._require_mlx().array(data, dtype=self.dtype)

    def _array_shape(self, tensor: Any) -> tuple[int, ...]:
        return tuple(int(x) for x in getattr(tensor, "shape", (len(tensor),)))

    def _array_dtype(self, tensor: Any) -> str:
        return str(getattr(tensor, "dtype", self.dtype))

    def _array_bytes(self, mx: Any, tensor: Any) -> bytes:
        contiguous = mx.ascontiguousarray(tensor)
        if hasattr(mx, "flatten"):
            flat = mx.flatten(contiguous)
        else:  # pragma: no cover - fallback for minimal runtimes
            flat = contiguous
        if hasattr(mx, "uint8"):
            byte_tensor = flat.astype(mx.uint8)
        else:  # pragma: no cover - fallback for minimal runtimes
            byte_tensor = flat
        values: list[int] = self._flatten_ints(byte_tensor.tolist())
        return bytes(value & 0xFF for value in values)

    def _array_from_bytes(self, data: bytes, dtype: str) -> Any:
        return self._require_mlx().array(list(data), dtype=dtype)

    @staticmethod
    def _flatten_ints(values: Any) -> list[int]:
        if isinstance(values, list):
            flattened: list[int] = []
            for value in values:
                flattened.extend(MLXTensorCodec._flatten_ints(value))
            return flattened
        return [int(values)]

    def _page_like(self, page: KVPage, *, data: bytes, codec_id: str, status: str) -> KVPage:
        return KVPage(
            page_id=page.page_id,
            data=data,
            data_hash=hash_bytes(data),
            token_offset=page.token_offset,
            token_count=page.token_count,
            layer_index=page.layer_index,
            head_range=page.head_range,
            codec_id=codec_id,
            status=status,
            created_at=page.created_at,
            actor=page.actor,
            action_id=page.action_id,
            event_id=page.event_id,
        )


__all__: list[str] = [
    "MLXTensorCodec",
    "MLXTensorMetadata",
    "MLXUnavailableError",
]
