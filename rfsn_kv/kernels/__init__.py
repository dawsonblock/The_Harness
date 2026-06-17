"""Metal/MLX kernels for fused compressed attention.

This module will contain the Metal and MLX kernel implementations for
fused compressed attention when hardware support is available in Phase 10.
For now, it provides protocol definitions and placeholder interfaces.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


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


__all__: list[str] = ["FusedAttentionKernel"]
