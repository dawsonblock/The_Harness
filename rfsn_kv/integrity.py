"""Integrity checks for persisted and in-memory KV pages.

Verifies that page data matches its stored content hash, detects
corruption in stored pages, and can repair hash mismatches when the
data is authoritative.
"""

from __future__ import annotations

from dataclasses import dataclass

from rfsn_kv.common import hash_bytes
from rfsn_kv.pages import KVPage
from rfsn_kv.types import PageId


@dataclass(frozen=True, slots=True)
class IntegrityReport:
    """Result of an integrity check over a set of pages.

    Attributes:
        checked: Number of pages checked.
        corrupted: IDs of pages where data_hash does not match actual hash.
        valid: Number of pages that passed the check.
    """

    checked: int
    corrupted: tuple[PageId, ...]
    valid: int


class IntegrityChecker:
    """Verifies page integrity via SHA-256 content hashes.

    Stateless — all operations are pure functions over page data.
    """

    def verify_page(self, page: KVPage) -> bool:
        """Return True if ``page.data_hash`` matches the SHA-256 of ``page.data``."""
        actual = hash_bytes(page.data)
        return actual == page.data_hash

    def verify_page_range(self, pages: tuple[KVPage, ...]) -> IntegrityReport:
        """Check integrity of a batch of pages.

        Returns an IntegrityReport listing any pages whose data_hash
        does not match the actual SHA-256 of their data.
        """
        corrupted: list[PageId] = []
        for page in pages:
            if not self.verify_page(page):
                corrupted.append(page.page_id)
        return IntegrityReport(
            checked=len(pages),
            corrupted=tuple(corrupted),
            valid=len(pages) - len(corrupted),
        )

    def repair_page(self, page: KVPage) -> KVPage:
        """Return a new page with ``data_hash`` recomputed from ``data``.

        Use this when the data is authoritative and the hash may have
        drifted due to storage corruption or version mismatch.
        """
        corrected_hash = hash_bytes(page.data)
        if corrected_hash == page.data_hash:
            return page  # no repair needed
        return KVPage(
            page_id=page.page_id,
            data=page.data,
            data_hash=corrected_hash,
            token_offset=page.token_offset,
            token_count=page.token_count,
            layer_index=page.layer_index,
            head_range=page.head_range,
            codec_id=page.codec_id,
            status=page.status,
            created_at=page.created_at,
            actor=page.actor,
            action_id=page.action_id,
            event_id=page.event_id,
        )
