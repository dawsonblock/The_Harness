"""Base protocol for KV codecs. (Skeleton)"""

from __future__ import annotations

from typing import Protocol


class KVCodec(Protocol):
    """Codec contract implemented by every page compression backend."""

    codec_id: str
    codec_version: int
