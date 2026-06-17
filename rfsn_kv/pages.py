"""Immutable KV page representation for physical cache storage.

Each page holds a contiguous range of token key-value data for a single
transformer layer. Pages are immutable: once created, a page is never
mutated. Compression codecs produce new pages with updated data and
codec identifiers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from rfsn_kv.common import CASStore, hash_bytes, utc_now
from rfsn_kv.types import ContentHash, ContentReference, LayerIndex, PageId


@dataclass(frozen=True, slots=True)
class KVPage:
    """An immutable block of KV-cache data for a contiguous token range.

    Attributes:
        page_id: Unique identifier for this page.
        data: Raw (possibly compressed) byte payload.
        data_hash: SHA-256 hash of ``data``, validated at construction.
        token_offset: Starting token position in the global KV sequence.
        token_count: Number of tokens this page covers.
        layer_index: Transformer layer this page belongs to.
        head_range: Which attention heads this page stores. Empty means all.
        codec_id: Identifier of the codec that produced ``data``
            (``"identity"`` for uncompressed).
        status: Lifecycle status of the page.
        created_at: UTC creation timestamp.
        actor: Entity that created this page.
        action_id: Action identifier for provenance tracking.
        event_id: Optional event identifier for provenance tracking.
        content_ref: Optional pointer to content stored in a CAS store.
            When set, ``data`` is empty and must be resolved via
            ``resolve_content(cas)``.
    """

    page_id: PageId
    data: bytes
    data_hash: ContentHash
    token_offset: int
    token_count: int
    layer_index: LayerIndex
    head_range: tuple[int, ...] = field(default_factory=tuple)
    codec_id: str = "identity"
    status: str = "decompressed"
    created_at: datetime = field(default_factory=utc_now)
    actor: str = "system"
    action_id: str = "init"
    event_id: str | None = None
    content_ref: ContentReference | None = None

    def __post_init__(self) -> None:
        if self.content_ref is None:
            expected = hash_bytes(self.data)
            if expected != self.data_hash:
                raise ValueError(
                    f"KVPage {self.page_id}: data_hash mismatch: "
                    f"expected {expected}, got {self.data_hash}"
                )
        if self.token_offset < 0:
            raise ValueError(
                f"KVPage {self.page_id}: token_offset must be >= 0, "
                f"got {self.token_offset}"
            )
        if self.token_count <= 0:
            raise ValueError(
                f"KVPage {self.page_id}: token_count must be > 0, "
                f"got {self.token_count}"
            )

    @property
    def compressed_size(self) -> int:
        """Size of the stored data in bytes."""
        return len(self.data)

    @property
    def token_end(self) -> int:
        """Exclusive end of the token range."""
        return self.token_offset + self.token_count

    @property
    def is_offloaded(self) -> bool:
        """Return True if this page's data is stored in a CAS store."""
        return self.content_ref is not None

    def resolve_content(self, cas: CASStore) -> bytes:
        """Retrieve the page data from a CAS store.

        If the page is not offloaded (``content_ref is None``), the inline
        ``data`` field is returned directly.
        """
        if self.content_ref is None:
            return self.data
        return cas.get_bytes(self.content_ref.content_hash)

    @classmethod
    def create(
        cls,
        *,
        page_id: PageId,
        data: bytes,
        token_offset: int,
        token_count: int,
        layer_index: LayerIndex,
        head_range: tuple[int, ...] | None = None,
        codec_id: str = "identity",
        status: str = "decompressed",
        actor: str = "system",
        action_id: str = "init",
        event_id: str | None = None,
    ) -> KVPage:
        """Create a page with a correctly computed data hash."""
        return cls(
            page_id=page_id,
            data=data,
            data_hash=hash_bytes(data),
            token_offset=token_offset,
            token_count=token_count,
            layer_index=layer_index,
            head_range=head_range or (),
            codec_id=codec_id,
            status=status,
            actor=actor,
            action_id=action_id,
            event_id=event_id,
        )

    @classmethod
    def create_with_cas(
        cls,
        *,
        page_id: PageId,
        data: bytes,
        token_offset: int,
        token_count: int,
        layer_index: LayerIndex,
        head_range: tuple[int, ...] | None = None,
        codec_id: str = "identity",
        status: str = "decompressed",
        actor: str = "system",
        action_id: str = "init",
        event_id: str | None = None,
        cas: CASStore,
    ) -> KVPage:
        """Create a page and offload its data to a CAS store.

        The page's ``data`` field is empty and ``content_ref`` points to
        the offloaded content. Use ``resolve_content(cas)`` to retrieve it.
        """
        data_hash = hash_bytes(data)
        cas.put(data)
        return cls(
            page_id=page_id,
            data=b"",
            data_hash=data_hash,
            token_offset=token_offset,
            token_count=token_count,
            layer_index=layer_index,
            head_range=head_range or (),
            codec_id=codec_id,
            status=status,
            actor=actor,
            action_id=action_id,
            event_id=event_id,
            content_ref=ContentReference(
                content_hash=data_hash,
                byte_length=len(data),
            ),
        )


@dataclass(frozen=True, slots=True)
class PageRange:
    """A contiguous range of pages by token position.

    Attributes:
        start_token: Inclusive start of the token range.
        end_token: Exclusive end of the token range.
        page_ids: Ordered page identifiers covering this range.
    """

    start_token: int
    end_token: int
    page_ids: tuple[PageId, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.end_token <= self.start_token:
            raise ValueError(
                f"PageRange: end_token ({self.end_token}) must be > "
                f"start_token ({self.start_token})"
            )
