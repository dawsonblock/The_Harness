"""Metal/MLX kernels for fused compressed attention.

This module will contain the Metal and MLX kernel implementations for
fused compressed attention when hardware support is available in Phase 10.
For now, it provides protocol definitions and placeholder interfaces.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class FusedAttentionKernel(Protocol):
    """Protocol for Metal/MLX fused compressed attention kernels.

    Kernels implementing this protocol must provide:
    - ``kernel_id``: Unique identifier for the kernel.
    - ``kernel_version``: Version number for compatibility tracking.
    - ``execute()``: Run the kernel with provided inputs.

    The actual ``execute()`` signature will be defined when MLX is available.
    """

    kernel_id: str
    kernel_version: int

    def get_info(self) -> dict[str, str | int]:
        """Return metadata about this kernel."""
        ...


@runtime_checkable
class MLXAttentionKernel(FusedAttentionKernel, Protocol):
    """Protocol for an MLX fused attention kernel."""

    def execute(
        self,
        *,
        q: Any,
        k: Any,
        v: Any,
        scale: float | None = None,
        cache: Any | None = None,
    ) -> Any:
        """Run fused attention on MLX arrays."""
        ...


@runtime_checkable
class CompressedPageAttentionKernel(FusedAttentionKernel, Protocol):
    """Protocol for attention over compressed KV page payloads."""

    def execute(
        self,
        *,
        query: Any,
        compressed_pages: Any,
        page_table: Any | None = None,
        codec_id: str | None = None,
    ) -> Any:
        """Run attention using compressed KV pages."""
        ...


__all__: list[str] = [
    "FusedAttentionKernel",
    "MLXAttentionKernel",
    "CompressedPageAttentionKernel",
]

