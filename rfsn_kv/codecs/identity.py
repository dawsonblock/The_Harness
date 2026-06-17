"""Identity (passthrough) codec — no compression.

Used as the default codec for uncompressed KV pages. Data passes through
unchanged; ``compress`` and ``decompress`` are no-ops that return new pages
with ``codec_id`` updated accordingly.
"""

from __future__ import annotations

from rfsn_kv.pages import KVPage


class IdentityCodec:
    """Passthrough codec that performs no compression.

    This is the default codec. ``compress`` returns a new page identical
    to the input except with ``codec_id = "identity"`` and
    ``status = "decompressed"``. ``decompress`` is a no-op alias.
    """

    codec_id: str = "identity"
    codec_version: int = 1

    def compress(self, page: KVPage) -> KVPage:
        """Return a new page with identity codec (no compression)."""
        return KVPage(
            page_id=page.page_id,
            data=page.data,
            data_hash=page.data_hash,
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

    def decompress(self, page: KVPage) -> KVPage:
        """Return a new page unchanged (identity is already uncompressed)."""
        return self.compress(page)

    def estimate_ratio(self, page: KVPage) -> float:
        """Ratio is always 1.0 — no compression."""
        return 1.0
