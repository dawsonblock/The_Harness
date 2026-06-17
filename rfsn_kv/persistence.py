"""SQLite persistence layer for encoded KV pages.

Stores compressed or uncompressed KV pages in a SQLite database with WAL
journaling. Provides atomic writes, idempotent puts, and efficient
retrieval by page ID or layer index.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from rfsn_kv.common import utc_now
from rfsn_kv.pages import KVPage
from rfsn_kv.types import ContentHash, LayerIndex, PageId

_CURRENT_SCHEMA_VERSION = 1


class KVPersistenceError(RuntimeError):
    """Raised for unexpected persistence failures."""


class IntegrityError(KVPersistenceError):
    """Raised when stored page data fails integrity validation."""


class KVPersistence:
    """SQLite-backed persistence for KV pages.

    Each page is stored as a row with its raw data, metadata, and a
    content hash for integrity verification. WAL journaling is used for
    concurrent read performance.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self.db_path),
            isolation_level=None,  # autocommit mode; explicit BEGIN/COMMIT
            detect_types=sqlite3.PARSE_DECLTYPES,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _cursor(self) -> sqlite3.Cursor:
        if self._conn is None:
            self._conn = self._connect()
        return self._conn.cursor()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> KVPersistence:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS kv_schema_version (
                        version INTEGER PRIMARY KEY
                    )
                    """
                )
                row = conn.execute(
                    "SELECT version FROM kv_schema_version ORDER BY version DESC LIMIT 1"
                ).fetchone()
                current_version = row["version"] if row else 0
                if current_version < _CURRENT_SCHEMA_VERSION:
                    self._apply_migration(conn, _CURRENT_SCHEMA_VERSION)
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def _apply_migration(self, conn: sqlite3.Connection, target_version: int) -> None:
        """Apply schema migrations up to target_version."""
        if target_version >= 1:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS kv_pages (
                    page_id TEXT PRIMARY KEY,
                    data BLOB NOT NULL,
                    data_hash TEXT NOT NULL,
                    token_offset INTEGER NOT NULL,
                    token_count INTEGER NOT NULL,
                    layer_index INTEGER NOT NULL,
                    head_range TEXT NOT NULL DEFAULT '[]',
                    codec_id TEXT NOT NULL DEFAULT 'identity',
                    status TEXT NOT NULL DEFAULT 'decompressed',
                    created_at TEXT NOT NULL,
                    actor TEXT NOT NULL DEFAULT 'system',
                    action_id TEXT NOT NULL DEFAULT 'init',
                    event_id TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_kv_pages_layer
                ON kv_pages (layer_index)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS kv_schema_version (
                    version INTEGER PRIMARY KEY
                )
                """
            )
            conn.execute(
                "INSERT OR REPLACE INTO kv_schema_version (version) VALUES (1)"
            )

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    def put_page(self, page: KVPage) -> None:
        """Insert or replace a page.

        If a page with the same ``page_id`` already exists, it is
        replaced only if the data hash matches (idempotent). A
        mismatch raises ``IntegrityError``.
        """
        cur = self._cursor()
        cur.execute("BEGIN IMMEDIATE")
        try:
            existing = cur.execute(
                "SELECT data_hash FROM kv_pages WHERE page_id = ?",
                (page.page_id,),
            ).fetchone()
            if existing is not None and existing["data_hash"] != page.data_hash:
                raise IntegrityError(
                    f"Page {page.page_id} already exists with different hash: "
                    f"existing={existing['data_hash']}, new={page.data_hash}"
                )
            cur.execute(
                """
                INSERT OR REPLACE INTO kv_pages
                    (page_id, data, data_hash, token_offset, token_count,
                     layer_index, head_range, codec_id, status,
                     created_at, actor, action_id, event_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    page.page_id,
                    page.data,
                    page.data_hash,
                    page.token_offset,
                    page.token_count,
                    page.layer_index,
                    json.dumps(list(page.head_range)),
                    page.codec_id,
                    page.status,
                    page.created_at.isoformat(),
                    page.actor,
                    page.action_id,
                    page.event_id,
                ),
            )
            cur.execute("COMMIT")
        except Exception:
            cur.execute("ROLLBACK")
            raise

    def get_page(self, page_id: PageId) -> KVPage:
        """Retrieve a page by its ID.

        Raises:
            KeyError: If no page with that ID exists.
        """
        cur = self._cursor()
        row = cur.execute(
            "SELECT * FROM kv_pages WHERE page_id = ?", (page_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Page {page_id!r} not found")
        return self._row_to_page(row)

    def has_page(self, page_id: PageId) -> bool:
        """Return True if a page with the given ID exists."""
        cur = self._cursor()
        row = cur.execute(
            "SELECT 1 FROM kv_pages WHERE page_id = ?", (page_id,)
        ).fetchone()
        result: bool = row is not None
        return result

    def delete_page(self, page_id: PageId) -> bool:
        """Delete a page by its ID. Returns True if a page was deleted."""
        cur = self._cursor()
        cur.execute("BEGIN IMMEDIATE")
        try:
            cur.execute("DELETE FROM kv_pages WHERE page_id = ?", (page_id,))
            changes_row = cur.execute("SELECT CHANGES()").fetchone()
            deleted: int = changes_row[0] if changes_row is not None else 0
            cur.execute("COMMIT")
            return deleted > 0
        except Exception:
            cur.execute("ROLLBACK")
            raise

    def list_pages(self) -> tuple[PageId, ...]:
        """Return all page IDs in the store."""
        cur = self._cursor()
        rows = cur.execute("SELECT page_id FROM kv_pages ORDER BY page_id").fetchall()
        return tuple(PageId(row["page_id"]) for row in rows)

    def list_pages_for_layer(self, layer_index: LayerIndex) -> tuple[PageId, ...]:
        """Return page IDs for a specific transformer layer."""
        cur = self._cursor()
        rows = cur.execute(
            "SELECT page_id FROM kv_pages WHERE layer_index = ? ORDER BY page_id",
            (layer_index,),
        ).fetchall()
        return tuple(PageId(row["page_id"]) for row in rows)

    def count_pages(self) -> int:
        """Return the total number of stored pages."""
        cur = self._cursor()
        row = cur.execute("SELECT COUNT(*) as cnt FROM kv_pages").fetchone()
        result: int = row["cnt"]
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _row_to_page(self, row: sqlite3.Row) -> KVPage:
        """Convert a database row to a KVPage instance."""
        head_range = tuple(json.loads(row["head_range"]))
        return KVPage(
            page_id=PageId(row["page_id"]),
            data=bytes(row["data"]),
            data_hash=ContentHash(row["data_hash"]),
            token_offset=row["token_offset"],
            token_count=row["token_count"],
            layer_index=LayerIndex(row["layer_index"]),
            head_range=head_range,
            codec_id=row["codec_id"],
            status=row["status"],
            created_at=utc_now(),
            actor=row["actor"],
            action_id=row["action_id"],
            event_id=row["event_id"],
        )
