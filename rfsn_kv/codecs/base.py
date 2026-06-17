"""Base protocol for KV codecs.

Every compression backend must implement this protocol. Codecs are used to
compress and decompress KV page data. The ``identity`` codec is the default
and stores data unchanged.
"""

from __future__ import annotations

from typing import Protocol

from rfsn_kv.pages import KVPage


class KVCodec(Protocol):
    """Codec contract implemented by every page compression backend."""

    codec_id: str
    codec_version: int

    def compress(self, page: KVPage) -> KVPage:
        """Return a new page with compressed data and updated codec_id."""
        ...

    def decompress(self, page: KVPage) -> KVPage:
        """Return a new page with uncompressed data and identity codec_id."""
        ...

    def estimate_ratio(self, page: KVPage) -> float:
        """Estimate compression ratio (compressed_size / original_size).

        Returns 1.0 for no compression, < 1.0 for actual compression.
        """
        ...
