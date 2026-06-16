"""SQLite WAL persistence and content-addressed filesystem store."""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from rfsn_agent.cas import ContentAddressedStore
from rfsn_agent.domain import BudgetLedger, HarnessSnapshot
from rfsn_agent.events import HarnessEvent
from rfsn_agent.reducer import reduce_event
from rfsn_agent.types import TrajectoryId

_CURRENT_STORE_SCHEMA_VERSION = 1


class StoreError(RuntimeError):
    """Raised for unexpected persistence failures."""


class ConcurrentAppendError(StoreError):
    """Raised when an optimistic concurrency check fails."""


class SchemaMigrationError(StoreError):
    """Raised when the database schema cannot be migrated to the target version."""


class SQLiteEventStore:
    """Append-only event store with WAL journaling and trajectory isolation.

    Each trajectory has an independent event sequence. Events are stored as
    JSON in SQLite, large objects can be offloaded to the CAS, and snapshots
    are reconstructed by replaying the event log through the pure reducer.
    """

    def __init__(
        self,
        db_path: str | Path,
        cas_base_dir: str | Path | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.cas = (
            ContentAddressedStore(cas_base_dir)
            if cas_base_dir is not None
            else None
        )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self.db_path),
            isolation_level=None,  # autocommit mode; we use explicit BEGIN/COMMIT
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

    def __enter__(self) -> SQLiteEventStore:
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
                    CREATE TABLE IF NOT EXISTS schema_version (
                        version INTEGER PRIMARY KEY
                    )
                    """
                )
                row = conn.execute(
                    "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
                ).fetchone()
                current_version = row["version"] if row else 0
                for version in range(current_version + 1, _CURRENT_STORE_SCHEMA_VERSION + 1):
                    _apply_migration(conn, version)
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    # ------------------------------------------------------------------
    # Trajectory lifecycle
    # ------------------------------------------------------------------

    def init_trajectory(
        self,
        trajectory_id: TrajectoryId,
        *,
        epoch_id: str = "epoch-0",
        budget: BudgetLedger | None = None,
    ) -> HarnessSnapshot:
        """Create a new trajectory with an empty initial snapshot."""
        if budget is None:
            budget = BudgetLedger(trajectory_id=trajectory_id, max_tokens=0)
        snapshot = HarnessSnapshot.create(
            trajectory_id=trajectory_id,
            epoch_id=epoch_id,
            sequence=0,
            budget=budget,
        )
        cur = self._cursor()
        cur.execute("BEGIN IMMEDIATE")
        try:
            cur.execute(
                """
                INSERT INTO trajectories (trajectory_id, created_at, epoch_id)
                VALUES (?, ?, ?)
                """,
                (trajectory_id, snapshot.created_at.isoformat(), epoch_id),
            )
            _insert_snapshot(cur, snapshot, event_count=0)
            cur.execute("COMMIT")
        except sqlite3.IntegrityError as exc:
            cur.execute("ROLLBACK")
            raise StoreError(
                f"Trajectory already exists: {trajectory_id}"
            ) from exc
        return snapshot

    def list_trajectories(self) -> list[str]:
        """Return all trajectory ids in creation order."""
        cur = self._cursor()
        rows = cur.execute(
            "SELECT trajectory_id FROM trajectories ORDER BY created_at"
        ).fetchall()
        return [row["trajectory_id"] for row in rows]

    # ------------------------------------------------------------------
    # Event append and fetch
    # ------------------------------------------------------------------

    def append_event(self, event: HarnessEvent) -> None:
        """Append a single event to its trajectory.

        Duplicate idempotency keys for the same trajectory are ignored if the
        stored event is byte-identical; mismatched duplicates raise.
        """
        cur = self._cursor()
        cur.execute("BEGIN IMMEDIATE")
        try:
            existing = self._load_event_by_idempotency(
                cur, event.trajectory_id, event.idempotency_key
            )
            if existing is not None:
                if existing.to_dict() == event.to_dict():
                    cur.execute("COMMIT")
                    return
                raise ConcurrentAppendError(
                    f"Idempotency key {event.idempotency_key} already used by a different event"
                )
            max_seq_row = cur.execute(
                "SELECT MAX(sequence) AS max_seq FROM harness_events WHERE trajectory_id = ?",
                (event.trajectory_id,),
            ).fetchone()
            max_seq = (
                max_seq_row["max_seq"]
                if max_seq_row and max_seq_row["max_seq"] is not None
                else 0
            )
            expected = max_seq + 1
            if event.sequence != expected:
                raise ConcurrentAppendError(
                    f"Event {event.event_id} has sequence {event.sequence}, expected {expected}"
                )
            _insert_event(cur, event)
            cur.execute("COMMIT")
        except sqlite3.IntegrityError as exc:
            cur.execute("ROLLBACK")
            raise ConcurrentAppendError(
                f"Event {event.event_id} conflicts with existing sequence or idempotency key"
            ) from exc
        except ConcurrentAppendError:
            cur.execute("ROLLBACK")
            raise

    def append_events(self, events: list[HarnessEvent]) -> None:
        """Atomically append multiple events in sequence order."""
        if not events:
            return
        cur = self._cursor()
        cur.execute("BEGIN IMMEDIATE")
        try:
            for event in events:
                _insert_event(cur, event)
            cur.execute("COMMIT")
        except Exception:
            cur.execute("ROLLBACK")
            raise

    def get_events(
        self,
        trajectory_id: str,
        *,
        after_sequence: int = -1,
        limit: int | None = None,
    ) -> list[HarnessEvent]:
        """Return events for ``trajectory_id`` with sequence > ``after_sequence``."""
        cur = self._cursor()
        sql = """
            SELECT payload_json FROM harness_events
            WHERE trajectory_id = ? AND sequence > ?
            ORDER BY sequence
        """
        params: list[Any] = [trajectory_id, after_sequence]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        rows = cur.execute(sql, params).fetchall()
        return [HarnessEvent.from_dict(json.loads(row["payload_json"])) for row in rows]

    def get_event_count(self, trajectory_id: str) -> int:
        cur = self._cursor()
        row = cur.execute(
            "SELECT COUNT(*) AS cnt FROM harness_events WHERE trajectory_id = ?",
            (trajectory_id,),
        ).fetchone()
        return int(row["cnt"])

    def _load_event_by_idempotency(
        self,
        cur: sqlite3.Cursor,
        trajectory_id: str,
        idempotency_key: str,
    ) -> HarnessEvent | None:
        row = cur.execute(
            """
            SELECT payload_json FROM harness_events
            WHERE trajectory_id = ? AND idempotency_key = ?
            """,
            (trajectory_id, idempotency_key),
        ).fetchone()
        if row is None:
            return None
        return HarnessEvent.from_dict(json.loads(row["payload_json"]))

    # ------------------------------------------------------------------
    # Replay and snapshots
    # ------------------------------------------------------------------

    def replay(
        self,
        trajectory_id: str,
        *,
        up_to_sequence: int | None = None,
    ) -> HarnessSnapshot:
        """Replay events for ``trajectory_id`` into a snapshot.

        If ``up_to_sequence`` is provided, replay only events with sequence
        <= that value.
        """
        cur = self._cursor()
        row = cur.execute(
            "SELECT * FROM trajectories WHERE trajectory_id = ?",
            (trajectory_id,),
        ).fetchone()
        if row is None:
            raise StoreError(f"Trajectory not found: {trajectory_id}")

        checkpoint_sql = """
            SELECT snapshot_json FROM harness_snapshots
            WHERE trajectory_id = ?
        """
        params: list[Any] = [trajectory_id]
        if up_to_sequence is not None:
            checkpoint_sql += " AND sequence <= ?"
            params.append(up_to_sequence)
        checkpoint_sql += " ORDER BY sequence DESC LIMIT 1"
        row = cur.execute(checkpoint_sql, params).fetchone()
        if row is None:
            raise StoreError(f"No snapshot checkpoint found for trajectory: {trajectory_id}")
        snapshot = HarnessSnapshot.from_dict(json.loads(row["snapshot_json"]))
        events = self.get_events(trajectory_id, after_sequence=snapshot.sequence)
        for event in events:
            if up_to_sequence is not None and event.sequence > up_to_sequence:
                break
            snapshot = reduce_event(snapshot, event)
        return snapshot

    def get_latest_snapshot(self, trajectory_id: str) -> HarnessSnapshot:
        """Return the latest snapshot by replaying the full event log."""
        return self.replay(trajectory_id)

    def save_snapshot(self, snapshot: HarnessSnapshot, event_count: int) -> None:
        """Persist a snapshot checkpoint for fast recovery."""
        cur = self._cursor()
        cur.execute("BEGIN IMMEDIATE")
        try:
            _insert_snapshot(cur, snapshot, event_count)
            cur.execute("COMMIT")
        except sqlite3.IntegrityError as exc:
            cur.execute("ROLLBACK")
            raise StoreError(
                f"Snapshot checkpoint conflict for {snapshot.trajectory_id}"
            ) from exc

    def get_latest_checkpoint_sequence(self, trajectory_id: str) -> int | None:
        """Return the sequence of the most recent saved snapshot checkpoint."""
        cur = self._cursor()
        row = cur.execute(
            """
            SELECT sequence FROM harness_snapshots
            WHERE trajectory_id = ?
            ORDER BY sequence DESC LIMIT 1
            """,
            (trajectory_id,),
        ).fetchone()
        return row["sequence"] if row else None

    # ------------------------------------------------------------------
    # CAS helpers
    # ------------------------------------------------------------------

    def put_blob(self, data: bytes | str) -> str:
        """Store a large object in the CAS and return its content hash."""
        if self.cas is None:
            raise StoreError("CAS base directory was not configured")
        return self.cas.put(data)

    def get_blob(self, content_hash: str) -> bytes:
        if self.cas is None:
            raise StoreError("CAS base directory was not configured")
        return self.cas.get(content_hash)


# ---------------------------------------------------------------------------
# Schema migrations
# ---------------------------------------------------------------------------


def _apply_migration(conn: sqlite3.Connection, version: int) -> None:
    if version == 1:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trajectories (
                trajectory_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                epoch_id TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS harness_events (
                event_id TEXT PRIMARY KEY,
                trajectory_id TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                schema_version INTEGER NOT NULL,
                idempotency_key TEXT NOT NULL,
                parent_event_id TEXT,
                created_at TEXT NOT NULL,
                actor TEXT NOT NULL,
                action_id TEXT NOT NULL,
                payload_hash TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                FOREIGN KEY (trajectory_id) REFERENCES trajectories(trajectory_id),
                UNIQUE(trajectory_id, sequence),
                UNIQUE(trajectory_id, idempotency_key)
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_harness_events_trajectory_sequence
            ON harness_events(trajectory_id, sequence)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS harness_snapshots (
                snapshot_id TEXT PRIMARY KEY,
                trajectory_id TEXT NOT NULL,
                epoch_id TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                state_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                event_count INTEGER NOT NULL,
                snapshot_json TEXT NOT NULL,
                FOREIGN KEY (trajectory_id) REFERENCES trajectories(trajectory_id),
                UNIQUE(trajectory_id, sequence)
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_harness_snapshots_trajectory_sequence
            ON harness_snapshots(trajectory_id, sequence)
            """
        )
    else:
        raise SchemaMigrationError(f"Unknown migration version: {version}")

    conn.execute(
        "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
        (version,),
    )


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _insert_event(cur: sqlite3.Cursor, event: HarnessEvent) -> None:
    cur.execute(
        """
        INSERT INTO harness_events (
            event_id, trajectory_id, sequence, event_type, schema_version,
            idempotency_key, parent_event_id, created_at, actor, action_id,
            payload_hash, payload_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event.event_id,
            event.trajectory_id,
            event.sequence,
            event.event_type,
            event.header.schema_version,
            event.idempotency_key,
            event.header.parent_event_id,
            event.header.created_at.isoformat(),
            event.header.actor,
            event.header.action_id,
            event.payload_hash,
            json.dumps(event.to_dict(), sort_keys=True),
        ),
    )


def _insert_snapshot(
    cur: sqlite3.Cursor, snapshot: HarnessSnapshot, event_count: int
) -> None:
    cur.execute(
        """
        INSERT OR REPLACE INTO harness_snapshots (
            snapshot_id, trajectory_id, epoch_id, sequence, state_hash,
            created_at, event_count, snapshot_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(uuid.uuid4()),
            snapshot.trajectory_id,
            snapshot.epoch_id,
            snapshot.sequence,
            snapshot.state_hash,
            snapshot.created_at.isoformat(),
            event_count,
            json.dumps(snapshot.to_dict(), sort_keys=True),
        ),
    )
