"""SQLite WAL persistence and content-addressed filesystem store."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from rfsn_agent.cas import ContentAddressedStore
from rfsn_agent.common import canonical_json, hash_content, utc_now
from rfsn_agent.domain import BudgetLedger, HarnessSnapshot
from rfsn_agent.events import (
    CURRENT_EVENT_SCHEMA_VERSION,
    EventHeader,
    HarnessEvent,
    ProposedEvent,
    compute_event_hash,
    compute_request_hash,
    compute_signature,
    verify_signature,
)
from rfsn_agent.reducer import reduce_event
from rfsn_agent.types import (
    ContentHash,
    EventId,
    TrajectoryId,
    VerificationResult,
    VerificationStatus,
)

_CURRENT_STORE_SCHEMA_VERSION = 3


class StoreError(RuntimeError):
    """Raised for unexpected persistence failures."""


class ConcurrentAppendError(StoreError):
    """Raised when an optimistic concurrency check fails."""


class StaleContextError(StoreError):
    """Raised when the trajectory head changed between read and commit."""


class IdempotencyConflictError(StoreError):
    """Raised when an idempotency key is reused with a different request."""


class SchemaMigrationError(StoreError):
    """Raised when the database schema cannot be migrated to the target version."""


class IntegrityError(StoreError):
    """Raised when stored event bytes fail integrity validation."""


@dataclass(frozen=True, slots=True)
class CommitReceipt:
    """A receipt for one committed event."""

    event_id: EventId
    sequence: int
    event_hash: ContentHash
    idempotency_key: str
    request_hash: ContentHash


@dataclass(frozen=True, slots=True)
class CommitResult:
    """Result of committing a batch of proposed events."""

    trajectory_id: TrajectoryId
    first_sequence: int
    last_sequence: int
    head_hash: ContentHash | None
    committed_events: tuple[HarnessEvent, ...]
    receipts: tuple[CommitReceipt, ...]


class SQLiteEventStore:
    """Append-only event store with WAL journaling and trajectory isolation.

    Each trajectory has an independent event sequence. Events are stored as
    JSON in SQLite, large objects can be offloaded to the CAS, and snapshots
    are reconstructed by replaying the event log through the pure reducer.

    Envelope metadata (sequence, timestamp, physical event_id, predecessor and
    event hashes) is owned by the store commit path, not by callers.
    """

    def __init__(
        self,
        db_path: str | Path,
        cas_base_dir: str | Path | None = None,
        *,
        signing_key: bytes | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.cas = (
            ContentAddressedStore(cas_base_dir)
            if cas_base_dir is not None
            else None
        )
        self.signing_key = signing_key
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._thread_local = threading.local()
        self._snapshot_cache: dict[tuple[str, int | None], HarnessSnapshot] = {}
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

    def _invalidate_snapshot_cache(self, trajectory_id: str) -> None:
        """Invalidate cached snapshots for a trajectory after mutations."""
        for key in list(self._snapshot_cache):
            if key[0] == trajectory_id:
                del self._snapshot_cache[key]

    def _cursor(self) -> sqlite3.Cursor:
        conn = getattr(self._thread_local, "conn", None)
        if conn is None:
            conn = self._connect()
            self._thread_local.conn = conn
        return conn.cursor()

    def _execute_in_thread_safe_connection(
        self, func: Callable[[sqlite3.Connection], Any]
    ) -> Any:
        """Run a SQLite operation in a fresh connection for async/threaded callers.

        The public store API is synchronous and may be used from one owning
        thread. Async callers such as ``ToolWorker`` use ``to_thread`` and can
        hit SQLite's thread affinity if they reuse ``_cursor``. This helper
        avoids that by opening a short-lived connection for the operation while
        leaving the normal connection cache unchanged.
        """
        with self._connect() as conn:
            return func(conn)

    def _get_latest_snapshot_thread_safe(self, trajectory_id: str) -> HarnessSnapshot:
        """Thread-safe latest snapshot path for async callers."""
        cache_key = (trajectory_id, None)
        cached = self._snapshot_cache.get(cache_key)
        if cached is not None:
            return cached

        def _load(conn: sqlite3.Connection) -> HarnessSnapshot:
            cur = conn.cursor()
            row = cur.execute(
                "SELECT * FROM trajectories WHERE trajectory_id = ?",
                (trajectory_id,),
            ).fetchone()
            if row is None:
                raise StoreError(f"Trajectory not found: {trajectory_id}")
            checkpoint_sql = """
                SELECT snapshot_json FROM harness_snapshots
                WHERE trajectory_id = ? ORDER BY sequence DESC LIMIT 1
            """
            row = cur.execute(checkpoint_sql, (trajectory_id,)).fetchone()
            if row is None:
                raise StoreError(
                    f"No snapshot checkpoint found for trajectory: {trajectory_id}"
                )
            snapshot = HarnessSnapshot.from_dict(json.loads(row["snapshot_json"]))

            event_sql = """
                SELECT
                    event_id, trajectory_id, sequence, event_type, schema_version,
                    idempotency_key, parent_event_id, created_at, actor, action_id,
                    payload_hash, previous_event_hash, event_hash,
                    previous_signature, signature, payload_json
                FROM harness_events
                WHERE trajectory_id = ? AND sequence > ?
                ORDER BY sequence
            """
            event_rows = cur.execute(
                event_sql, (trajectory_id, snapshot.sequence)
            ).fetchall()
            for event_row in event_rows:
                event = _event_from_row(event_row, self.signing_key, self.cas)
                snapshot = reduce_event(snapshot, event)
            return snapshot

        snapshot = self._execute_in_thread_safe_connection(_load)
        self._snapshot_cache[cache_key] = snapshot
        return snapshot

    def close(self) -> None:
        conn = getattr(self._thread_local, "conn", None)
        if conn is not None:
            conn.close()
            self._thread_local.conn = None

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
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS schema_lock (
                        lock_id INTEGER PRIMARY KEY CHECK (lock_id = 1),
                        acquired_at TEXT NOT NULL
                    )
                    """
                )
                row = conn.execute(
                    "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
                ).fetchone()
                current_version = row["version"] if row else 0
                for version in range(current_version + 1, _CURRENT_STORE_SCHEMA_VERSION + 1):
                    _apply_migration(conn, version, self.signing_key)
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
            last_event_hash=None,
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
    # Transactional commit
    # ------------------------------------------------------------------

    def commit_events(
        self,
        *,
        trajectory_id: TrajectoryId,
        expected_sequence: int,
        expected_head_hash: ContentHash | None,
        proposed_events: tuple[ProposedEvent, ...],
    ) -> CommitResult:
        """Atomically commit proposed events to a trajectory.

        The store validates the optimistic lock (``expected_sequence`` and
        ``expected_head_hash``), assigns physical envelope fields, resolves
        idempotency, reduces each event, and inserts the batch under a single
        SQLite transaction.
        """
        if not proposed_events:
            raise StoreError("commit_events called with no proposed events")

        cur = self._cursor()
        cur.execute("BEGIN IMMEDIATE")
        try:
            head = self._load_trajectory_head(cur, trajectory_id)
            if head.sequence != expected_sequence:
                raise StaleContextError(
                    f"Trajectory head changed: expected sequence {expected_sequence}, "
                    f"got {head.sequence}"
                )
            if head.last_event_hash != expected_head_hash:
                raise StaleContextError(
                    f"Trajectory head changed: expected head hash {expected_head_hash}, "
                    f"got {head.last_event_hash}"
                )

            committed: list[HarnessEvent] = []
            receipts: list[CommitReceipt] = []
            next_sequence = head.sequence + 1
            previous_hash = head.last_event_hash
            previous_signature = head.last_signature

            for proposed in proposed_events:
                request_hash = compute_request_hash(proposed)
                existing = self._load_idempotency_entry(
                    cur, trajectory_id, proposed.idempotency_key
                )
                if existing is not None:
                    if existing.request_hash != request_hash:
                        raise IdempotencyConflictError(
                            f"Idempotency key {proposed.idempotency_key!r} already used "
                            "by a different request"
                        )
                    # Same logical request: return the original receipt without
                    # mutating state. The caller is responsible for reconciling
                    # sequence gaps if the original commit advanced the head.
                    receipts.append(existing.receipt)
                    continue

                event_id = EventId(uuid.uuid4().hex)
                created_at = utc_now()
                payload_hash = hash_content(canonical_json(_payload_to_dict(proposed.payload)))
                header = EventHeader(
                    event_id=event_id,
                    trajectory_id=trajectory_id,
                    sequence=next_sequence,
                    event_type=proposed.event_type,
                    schema_version=CURRENT_EVENT_SCHEMA_VERSION,
                    idempotency_key=proposed.idempotency_key,
                    parent_event_id=proposed.parent_event_id,
                    created_at=created_at,
                    actor=proposed.actor,
                    action_id=proposed.action_id,
                    previous_event_hash=previous_hash,
                    previous_signature=previous_signature,
                    event_hash=ContentHash(""),
                    signature=None,
                )
                event_hash = compute_event_hash(header, payload_hash)
                object.__setattr__(header, "event_hash", event_hash)
                signature = compute_signature(header, self.signing_key)
                object.__setattr__(header, "signature", signature)
                event = HarnessEvent(
                    header=header, payload=proposed.payload, payload_hash=payload_hash
                )

                head = reduce_event(head, event)
                _insert_event(cur, event, request_hash, self.cas)
                committed.append(event)
                receipt = CommitReceipt(
                    event_id=event_id,
                    sequence=next_sequence,
                    event_hash=event_hash,
                    idempotency_key=proposed.idempotency_key,
                    request_hash=request_hash,
                )
                receipts.append(receipt)
                self._upsert_idempotency_entry(
                    cur, trajectory_id, proposed.idempotency_key, request_hash, receipt
                )

                next_sequence += 1
                previous_hash = event_hash
                previous_signature = signature

            if head.sequence > 0 and head.sequence % 50 == 0:
                event_count = self.get_event_count(trajectory_id)
                _insert_snapshot(cur, head, event_count)

            cur.execute("COMMIT")
            self._invalidate_snapshot_cache(trajectory_id)
        except (StaleContextError, IdempotencyConflictError):
            cur.execute("ROLLBACK")
            raise
        except Exception:
            cur.execute("ROLLBACK")
            raise StoreError("Failed to commit events") from None

        return CommitResult(
            trajectory_id=trajectory_id,
            first_sequence=committed[0].sequence if committed else head.sequence,
            last_sequence=head.sequence,
            head_hash=head.last_event_hash,
            committed_events=tuple(committed),
            receipts=tuple(receipts),
        )

    def _load_trajectory_head(
        self, cur: sqlite3.Cursor, trajectory_id: str
    ) -> HarnessSnapshot:
        """Load the authoritative head for a trajectory by replaying the log.

        Because snapshot checkpoints are optional and may lag behind the event
        log, the head is reconstructed from the latest checkpoint plus all
        events committed after it.
        """
        row = cur.execute(
            "SELECT * FROM trajectories WHERE trajectory_id = ?",
            (trajectory_id,),
        ).fetchone()
        if row is None:
            raise StoreError(f"Trajectory not found: {trajectory_id}")

        checkpoint_sql = """
            SELECT snapshot_json FROM harness_snapshots
            WHERE trajectory_id = ? ORDER BY sequence DESC LIMIT 1
        """
        row = cur.execute(checkpoint_sql, (trajectory_id,)).fetchone()
        if row is None:
            raise StoreError(f"No snapshot checkpoint found for trajectory: {trajectory_id}")
        snapshot = HarnessSnapshot.from_dict(json.loads(row["snapshot_json"]))

        event_sql = """
            SELECT
                event_id, trajectory_id, sequence, event_type, schema_version,
                idempotency_key, parent_event_id, created_at, actor, action_id,
                payload_hash, previous_event_hash, event_hash,
                previous_signature, signature, payload_json
            FROM harness_events
            WHERE trajectory_id = ? AND sequence > ?
            ORDER BY sequence
        """
        event_rows = cur.execute(event_sql, (trajectory_id, snapshot.sequence)).fetchall()
        for event_row in event_rows:
            event = _event_from_row(event_row, self.signing_key, self.cas)
            snapshot = reduce_event(snapshot, event)
        return snapshot

    def _load_idempotency_entry(
        self,
        cur: sqlite3.Cursor,
        trajectory_id: str,
        idempotency_key: str,
    ) -> _IdempotencyEntry | None:
        row = cur.execute(
            """
            SELECT request_hash, event_id, sequence, event_hash
            FROM idempotency_keys
            WHERE trajectory_id = ? AND idempotency_key = ?
            """,
            (trajectory_id, idempotency_key),
        ).fetchone()
        if row is None:
            return None
        return _IdempotencyEntry(
            request_hash=ContentHash(row["request_hash"]),
            receipt=CommitReceipt(
                event_id=EventId(row["event_id"]),
                sequence=int(row["sequence"]),
                event_hash=ContentHash(row["event_hash"]),
                idempotency_key=idempotency_key,
                request_hash=ContentHash(row["request_hash"]),
            ),
        )

    def _upsert_idempotency_entry(
        self,
        cur: sqlite3.Cursor,
        trajectory_id: str,
        idempotency_key: str,
        request_hash: ContentHash,
        receipt: CommitReceipt,
    ) -> None:
        cur.execute(
            """
            INSERT OR REPLACE INTO idempotency_keys
            (trajectory_id, idempotency_key, request_hash, event_id, sequence, event_hash)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                trajectory_id,
                idempotency_key,
                request_hash,
                receipt.event_id,
                receipt.sequence,
                receipt.event_hash,
            ),
        )

    # ------------------------------------------------------------------
    # Event fetch
    # ------------------------------------------------------------------

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
            SELECT
                event_id, trajectory_id, sequence, event_type, schema_version,
                idempotency_key, parent_event_id, created_at, actor, action_id,
                payload_hash, previous_event_hash, event_hash,
                previous_signature, signature, payload_json
            FROM harness_events
            WHERE trajectory_id = ? AND sequence > ?
            ORDER BY sequence
        """
        params: list[Any] = [trajectory_id, after_sequence]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        rows = cur.execute(sql, params).fetchall()
        return [_event_from_row(row, self.signing_key, self.cas) for row in rows]

    def get_event_count(self, trajectory_id: str) -> int:
        cur = self._cursor()
        row = cur.execute(
            "SELECT COUNT(*) AS cnt FROM harness_events WHERE trajectory_id = ?",
            (trajectory_id,),
        ).fetchone()
        return int(row["cnt"])

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
        """Return the latest snapshot, using an in-memory head cache when safe."""
        cache_key = (trajectory_id, None)
        cached = self._snapshot_cache.get(cache_key)
        if cached is not None:
            return cached
        snapshot = self.replay(trajectory_id)
        self._snapshot_cache[cache_key] = snapshot
        return snapshot

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


@dataclass(frozen=True, slots=True)
class _IdempotencyEntry:
    request_hash: ContentHash
    receipt: CommitReceipt


def _apply_migration(
    conn: sqlite3.Connection, version: int, signing_key: bytes | None
) -> None:
    if version == 1:
        _apply_v1_migration(conn)
    elif version == 2:
        _apply_v2_migration(conn)
    elif version == 3:
        _apply_v3_migration(conn, signing_key)
    else:
        raise SchemaMigrationError(f"Unknown migration version: {version}")

    conn.execute(
        "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
        (version,),
    )


def _apply_v1_migration(conn: sqlite3.Connection) -> None:
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


def _apply_v2_migration(conn: sqlite3.Connection) -> None:
    """Backfill hash-chain fields and authoritative envelope columns.

    Existing v1 events are rewritten with ``previous_event_hash`` and
    ``event_hash``; snapshots gain ``last_event_hash`` and have their state
    hashes recomputed. The migration validates a full replay before committing.
    """
    conn.execute("ALTER TABLE harness_events ADD COLUMN request_hash TEXT")
    conn.execute("ALTER TABLE harness_events ADD COLUMN previous_event_hash TEXT")
    conn.execute("ALTER TABLE harness_events ADD COLUMN event_hash TEXT")
    conn.execute("ALTER TABLE harness_snapshots ADD COLUMN last_event_hash TEXT")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS idempotency_keys (
            trajectory_id TEXT NOT NULL,
            idempotency_key TEXT NOT NULL,
            request_hash TEXT NOT NULL,
            event_id TEXT NOT NULL,
            sequence INTEGER NOT NULL,
            event_hash TEXT NOT NULL,
            PRIMARY KEY (trajectory_id, idempotency_key),
            FOREIGN KEY (trajectory_id) REFERENCES trajectories(trajectory_id)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_idempotency_keys_trajectory
        ON idempotency_keys(trajectory_id)
        """
    )

    _backfill_v2_hashes(conn)

    # Rebuild tables with strict non-null constraints.
    conn.execute(
        """
        CREATE TABLE harness_events_new (
            event_id TEXT PRIMARY KEY,
            trajectory_id TEXT NOT NULL,
            sequence INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            schema_version INTEGER NOT NULL,
            idempotency_key TEXT NOT NULL,
            request_hash TEXT NOT NULL,
            parent_event_id TEXT,
            created_at TEXT NOT NULL,
            actor TEXT NOT NULL,
            action_id TEXT NOT NULL,
            payload_hash TEXT NOT NULL,
            previous_event_hash TEXT,
            event_hash TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            FOREIGN KEY (trajectory_id) REFERENCES trajectories(trajectory_id),
            UNIQUE(trajectory_id, sequence),
            UNIQUE(trajectory_id, idempotency_key)
        )
        """
    )
    conn.execute(
        """
        INSERT INTO harness_events_new
        SELECT
            event_id, trajectory_id, sequence, event_type, schema_version,
            idempotency_key, request_hash, parent_event_id, created_at, actor,
            action_id, payload_hash, previous_event_hash, event_hash, payload_json
        FROM harness_events
        """
    )
    conn.execute("DROP TABLE harness_events")
    conn.execute("ALTER TABLE harness_events_new RENAME TO harness_events")
    conn.execute(
        """
        CREATE INDEX idx_harness_events_trajectory_sequence
        ON harness_events(trajectory_id, sequence)
        """
    )

    conn.execute(
        """
        CREATE TABLE harness_snapshots_new (
            snapshot_id TEXT PRIMARY KEY,
            trajectory_id TEXT NOT NULL,
            epoch_id TEXT NOT NULL,
            sequence INTEGER NOT NULL,
            state_hash TEXT NOT NULL,
            last_event_hash TEXT,
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
        INSERT INTO harness_snapshots_new
        SELECT snapshot_id, trajectory_id, epoch_id, sequence, state_hash,
               last_event_hash, created_at, event_count, snapshot_json
        FROM harness_snapshots
        """
    )
    conn.execute("DROP TABLE harness_snapshots")
    conn.execute("ALTER TABLE harness_snapshots_new RENAME TO harness_snapshots")
    conn.execute(
        """
        CREATE INDEX idx_harness_snapshots_trajectory_sequence
        ON harness_snapshots(trajectory_id, sequence)
        """
    )


def _backfill_v2_hashes(conn: sqlite3.Connection) -> None:
    """Backfill hashes for all existing v1 events and snapshots."""
    trajectory_rows = conn.execute(
        "SELECT trajectory_id FROM trajectories ORDER BY created_at"
    ).fetchall()

    for traj_row in trajectory_rows:
        trajectory_id = traj_row["trajectory_id"]
        event_rows = conn.execute(
            """
            SELECT event_id, payload_json FROM harness_events
            WHERE trajectory_id = ? ORDER BY sequence
            """,
            (trajectory_id,),
        ).fetchall()

        previous_hash: ContentHash | None = None
        for row in event_rows:
            v1_data = json.loads(row["payload_json"])
            event = _event_from_v1_dict(v1_data)
            request_hash = compute_request_hash(
                ProposedEvent(
                    event_type=event.event_type,
                    payload=event.payload,
                    idempotency_key=event.idempotency_key,
                    actor=event.header.actor,
                    action_id=event.header.action_id,
                    parent_event_id=event.header.parent_event_id,
                )
            )
            event_hash = event.header.event_hash
            conn.execute(
                """
                UPDATE harness_events
                SET schema_version = ?,
                    request_hash = ?,
                    payload_hash = ?,
                    previous_event_hash = ?,
                    event_hash = ?,
                    payload_json = ?
                WHERE event_id = ?
                """,
                (
                    CURRENT_EVENT_SCHEMA_VERSION,
                    request_hash,
                    event.payload_hash,
                    previous_hash,
                    event_hash,
                    json.dumps(event.to_dict(), sort_keys=True),
                    row["event_id"],
                ),
            )
            previous_hash = event_hash

        # Backfill snapshot last_event_hash and recompute state hash.
        snapshot_rows = conn.execute(
            """
            SELECT snapshot_id, snapshot_json FROM harness_snapshots
            WHERE trajectory_id = ? ORDER BY sequence
            """,
            (trajectory_id,),
        ).fetchall()

        event_hash_by_sequence: dict[int, ContentHash] = {0: None}  # type: ignore[dict-item]
        for idx, row in enumerate(event_rows, start=1):
            # Sequence numbers are 1-based after the initial snapshot.
            event_hash_by_sequence[idx] = ContentHash(
                conn.execute(
                    "SELECT event_hash FROM harness_events WHERE event_id = ?",
                    (row["event_id"],),
                ).fetchone()["event_hash"]
            )

        for row in snapshot_rows:
            snapshot_data = json.loads(row["snapshot_json"])
            snapshot = _snapshot_from_v1_dict(snapshot_data)
            seq = snapshot.sequence
            last_event_hash = event_hash_by_sequence.get(seq)
            snapshot = HarnessSnapshot.create(
                trajectory_id=snapshot.trajectory_id,
                epoch_id=snapshot.epoch_id,
                sequence=seq,
                last_event_hash=last_event_hash,
                created_at=snapshot.created_at,
                candidates=snapshot.candidates,
                curated_items=snapshot.curated_items,
                claims=snapshot.claims,
                evidence_links=snapshot.evidence_links,
                verification_records=snapshot.verification_records,
                tasks=snapshot.tasks,
                budget=snapshot.budget,
                submissions=snapshot.submissions,
                tool_invocations=snapshot.tool_invocations,
                tool_results=snapshot.tool_results,
            )
            conn.execute(
                """
                UPDATE harness_snapshots
                SET last_event_hash = ?, state_hash = ?, snapshot_json = ?
                WHERE snapshot_id = ?
                """,
                (
                    snapshot.last_event_hash,
                    snapshot.state_hash,
                    json.dumps(snapshot.to_dict(), sort_keys=True),
                    row["snapshot_id"],
                ),
            )

    # Validate full replay for every trajectory.
    for traj_row in trajectory_rows:
        trajectory_id = traj_row["trajectory_id"]
        checkpoint_row = conn.execute(
            """
            SELECT snapshot_json FROM harness_snapshots
            WHERE trajectory_id = ? ORDER BY sequence DESC LIMIT 1
            """,
            (trajectory_id,),
        ).fetchone()
        if checkpoint_row is None:
            continue
        snapshot = HarnessSnapshot.from_dict(json.loads(checkpoint_row["snapshot_json"]))
        event_rows = conn.execute(
            """
            SELECT payload_json FROM harness_events
            WHERE trajectory_id = ? ORDER BY sequence
            """,
            (trajectory_id,),
        ).fetchall()

        expected_previous_hash: ContentHash | None = None
        for row in event_rows:
            event = HarnessEvent.from_dict(json.loads(row["payload_json"]))
            if event.header.previous_event_hash != expected_previous_hash:
                raise SchemaMigrationError(
                    f"Migration produced invalid chain for {trajectory_id}: "
                    f"event {event.sequence} previous hash mismatch"
                )
            snapshot = reduce_event(snapshot, event)
            expected_previous_hash = event.header.event_hash


def _apply_v3_migration(
    conn: sqlite3.Connection, signing_key: bytes | None
) -> None:
    """Add HMAC-SHA-256 signature chain columns and rewrite the chain.

    Because the event hash covers ``previous_signature``, adding signatures
    changes every event hash. This migration recomputes the full hash and
    signature chain for each trajectory and updates snapshots accordingly.
    When no ``signing_key`` is configured the chain is rewritten with null
    signatures so the schema is uniform.
    """
    conn.execute("ALTER TABLE harness_events ADD COLUMN previous_signature TEXT")
    conn.execute("ALTER TABLE harness_events ADD COLUMN signature TEXT")
    conn.execute("ALTER TABLE harness_snapshots ADD COLUMN last_signature TEXT")

    trajectory_rows = conn.execute(
        "SELECT trajectory_id FROM trajectories ORDER BY created_at"
    ).fetchall()

    for traj_row in trajectory_rows:
        trajectory_id = traj_row["trajectory_id"]
        event_rows = conn.execute(
            """
            SELECT
                event_id, trajectory_id, sequence, event_type, schema_version,
                idempotency_key, parent_event_id, created_at, actor, action_id,
                payload_hash, previous_event_hash, event_hash,
                previous_signature, signature, payload_json
            FROM harness_events
            WHERE trajectory_id = ? ORDER BY sequence
            """,
            (trajectory_id,),
        ).fetchall()

        previous_hash: ContentHash | None = None
        previous_signature: ContentHash | None = None
        hash_by_sequence: dict[int, ContentHash] = {}
        signature_by_sequence: dict[int, ContentHash | None] = {0: None}
        for idx, row in enumerate(event_rows, start=1):
            event = _event_from_row(row, None)
            new_header = EventHeader(
                event_id=event.header.event_id,
                trajectory_id=event.header.trajectory_id,
                sequence=event.header.sequence,
                event_type=event.header.event_type,
                schema_version=CURRENT_EVENT_SCHEMA_VERSION,
                idempotency_key=event.header.idempotency_key,
                parent_event_id=event.header.parent_event_id,
                created_at=event.header.created_at,
                actor=event.header.actor,
                action_id=event.header.action_id,
                previous_event_hash=previous_hash,
                event_hash=ContentHash(""),
                previous_signature=previous_signature,
                signature=None,
            )
            event_hash = compute_event_hash(new_header, event.payload_hash)
            object.__setattr__(new_header, "event_hash", event_hash)
            signature = compute_signature(new_header, signing_key)
            object.__setattr__(new_header, "signature", signature)
            new_event = HarnessEvent(
                header=new_header, payload=event.payload, payload_hash=event.payload_hash
            )

            conn.execute(
                """
                UPDATE harness_events
                SET previous_event_hash = ?,
                    event_hash = ?,
                    previous_signature = ?,
                    signature = ?,
                    payload_json = ?
                WHERE event_id = ?
                """,
                (
                    previous_hash,
                    event_hash,
                    previous_signature,
                    signature,
                    json.dumps(new_event.to_dict(), sort_keys=True),
                    row["event_id"],
                ),
            )
            conn.execute(
                """
                UPDATE idempotency_keys
                SET event_hash = ?
                WHERE trajectory_id = ? AND idempotency_key = ?
                """,
                (event_hash, trajectory_id, event.idempotency_key),
            )

            hash_by_sequence[idx] = event_hash
            signature_by_sequence[idx] = signature
            previous_hash = event_hash
            previous_signature = signature

        # Rewrite snapshots with recomputed last_event_hash, last_signature,
        # and state hash. The last event hash at sequence N is the hash of the
        # event whose sequence equals N.
        snapshot_rows = conn.execute(
            """
            SELECT snapshot_id, snapshot_json FROM harness_snapshots
            WHERE trajectory_id = ? ORDER BY sequence
            """,
            (trajectory_id,),
        ).fetchall()
        for row in snapshot_rows:
            snapshot = HarnessSnapshot.from_dict(json.loads(row["snapshot_json"]))
            seq = snapshot.sequence
            last_event_hash = hash_by_sequence.get(seq)
            last_signature = signature_by_sequence.get(seq)
            snapshot = HarnessSnapshot.create(
                trajectory_id=snapshot.trajectory_id,
                epoch_id=snapshot.epoch_id,
                sequence=seq,
                last_event_hash=last_event_hash,
                last_signature=last_signature,
                created_at=snapshot.created_at,
                candidates=snapshot.candidates,
                curated_items=snapshot.curated_items,
                claims=snapshot.claims,
                evidence_links=snapshot.evidence_links,
                verification_records=snapshot.verification_records,
                tasks=snapshot.tasks,
                budget=snapshot.budget,
                submissions=snapshot.submissions,
                tool_invocations=snapshot.tool_invocations,
                tool_results=snapshot.tool_results,
            )
            conn.execute(
                """
                UPDATE harness_snapshots
                SET last_event_hash = ?,
                    last_signature = ?,
                    state_hash = ?,
                    snapshot_json = ?
                WHERE snapshot_id = ?
                """,
                (
                    snapshot.last_event_hash,
                    snapshot.last_signature,
                    snapshot.state_hash,
                    json.dumps(snapshot.to_dict(), sort_keys=True),
                    row["snapshot_id"],
                ),
            )

    # Validate full replay for every trajectory.
    for traj_row in trajectory_rows:
        trajectory_id = traj_row["trajectory_id"]
        checkpoint_row = conn.execute(
            """
            SELECT snapshot_json FROM harness_snapshots
            WHERE trajectory_id = ? ORDER BY sequence DESC LIMIT 1
            """,
            (trajectory_id,),
        ).fetchone()
        if checkpoint_row is None:
            continue
        snapshot = HarnessSnapshot.from_dict(json.loads(checkpoint_row["snapshot_json"]))
        event_rows = conn.execute(
            """
            SELECT
                event_id, trajectory_id, sequence, event_type, schema_version,
                idempotency_key, parent_event_id, created_at, actor, action_id,
                payload_hash, previous_event_hash, event_hash,
                previous_signature, signature, payload_json
            FROM harness_events
            WHERE trajectory_id = ? ORDER BY sequence
            """,
            (trajectory_id,),
        ).fetchall()

        expected_previous_hash: ContentHash | None = None
        expected_previous_signature: ContentHash | None = None
        for row in event_rows:
            event = _event_from_row(row, signing_key)
            if event.header.previous_event_hash != expected_previous_hash:
                raise SchemaMigrationError(
                    f"Migration produced invalid hash chain for {trajectory_id}: "
                    f"event {event.sequence} previous hash mismatch"
                )
            if event.header.previous_signature != expected_previous_signature:
                raise SchemaMigrationError(
                    f"Migration produced invalid signature chain for {trajectory_id}: "
                    f"event {event.sequence} previous signature mismatch"
                )
            snapshot = reduce_event(snapshot, event)
            expected_previous_hash = event.header.event_hash
            expected_previous_signature = event.header.signature


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _insert_event(
    cur: sqlite3.Cursor,
    event: HarnessEvent,
    request_hash: ContentHash,
    cas: ContentAddressedStore | None = None,
) -> None:
    event_dict = event.to_dict()
    if cas is not None:
        event_dict["payload"] = _maybe_offload_to_cas(event_dict["payload"], cas)
    payload_json = json.dumps(event_dict, sort_keys=True)
    cur.execute(
        """
        INSERT INTO harness_events (
            event_id, trajectory_id, sequence, event_type, schema_version,
            idempotency_key, request_hash, parent_event_id, created_at, actor,
            action_id, payload_hash, previous_event_hash, event_hash,
            previous_signature, signature, payload_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event.event_id,
            event.trajectory_id,
            event.sequence,
            event.event_type,
            event.header.schema_version,
            event.idempotency_key,
            request_hash,
            event.header.parent_event_id,
            event.header.created_at.isoformat(),
            event.header.actor,
            event.header.action_id,
            event.payload_hash,
            event.header.previous_event_hash,
            event.header.event_hash,
            event.header.previous_signature,
            event.header.signature,
            payload_json,
        ),
    )


def _insert_snapshot(
    cur: sqlite3.Cursor, snapshot: HarnessSnapshot, event_count: int
) -> None:
    cur.execute(
        """
        INSERT OR REPLACE INTO harness_snapshots (
            snapshot_id, trajectory_id, epoch_id, sequence, state_hash,
            last_event_hash, created_at, event_count, snapshot_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(uuid.uuid4()),
            snapshot.trajectory_id,
            snapshot.epoch_id,
            snapshot.sequence,
            snapshot.state_hash,
            snapshot.last_event_hash,
            snapshot.created_at.isoformat(),
            event_count,
            json.dumps(snapshot.to_dict(), sort_keys=True),
        ),
    )


def _event_from_row(
    row: sqlite3.Row,
    signing_key: bytes | None,
    cas: ContentAddressedStore | None = None,
) -> HarnessEvent:
    """Reconstruct a HarnessEvent from SQL columns and payload JSON.

    Every duplicated SQL field is cross-checked against the serialized JSON,
    and signatures are verified when a ``signing_key`` is configured.
    """
    data = json.loads(row["payload_json"])
    if cas is not None:
        data["payload"] = _maybe_resolve_from_cas(data["payload"], cas)
    header = data["header"]

    sql_fields = {
        "event_id": row["event_id"],
        "trajectory_id": row["trajectory_id"],
        "sequence": row["sequence"],
        "event_type": row["event_type"],
        "schema_version": row["schema_version"],
        "idempotency_key": row["idempotency_key"],
        "parent_event_id": row["parent_event_id"],
        "created_at": row["created_at"],
        "actor": row["actor"],
        "action_id": row["action_id"],
        "previous_event_hash": row["previous_event_hash"],
        "event_hash": row["event_hash"],
        "previous_signature": row["previous_signature"],
        "signature": row["signature"],
    }
    json_fields = {
        "event_id": header.get("event_id"),
        "trajectory_id": header.get("trajectory_id"),
        "sequence": header.get("sequence"),
        "event_type": header.get("event_type"),
        "schema_version": header.get("schema_version"),
        "idempotency_key": header.get("idempotency_key"),
        "parent_event_id": header.get("parent_event_id"),
        "created_at": header.get("created_at"),
        "actor": header.get("actor"),
        "action_id": header.get("action_id"),
        "previous_event_hash": header.get("previous_event_hash"),
        "event_hash": header.get("event_hash"),
        "previous_signature": header.get("previous_signature"),
        "signature": header.get("signature"),
    }
    if sql_fields != json_fields:
        raise IntegrityError(
            f"Event {row['event_id']} SQL metadata disagrees with serialized payload"
        )

    if data["payload_hash"] != row["payload_hash"]:
        raise IntegrityError(
            f"Event {row['event_id']} payload_hash SQL/JSON disagreement"
        )

    event = HarnessEvent.from_dict(data)
    try:
        verify_signature(event.header, signing_key)
    except ValueError as exc:
        raise IntegrityError(
            f"Event {row['event_id']} signature verification failed"
        ) from exc
    return event


def _maybe_offload_to_cas(obj: Any, cas: ContentAddressedStore | None) -> Any:
    """Offload large strings to CAS using iterative traversal.

    This avoids recursion errors for deeply nested payloads while preserving the
    ``{"__cas_ref__": hash}`` marker contract used by CAS-backed events.
    """
    if cas is None:
        return obj

    root: Any = obj
    stack: list[tuple[Any, tuple[Any, ...] | None, str | None]] = [(obj, None, None)]

    while stack:
        current, parent_items, key_in_parent = stack.pop()

        if isinstance(current, str):
            replacement: Any = (
                {"__cas_ref__": cas.put(current)} if len(current) > 4096 else current
            )
            if parent_items is not None and key_in_parent is not None:
                container, parent_key = parent_items, key_in_parent
                if isinstance(container, list):
                    container[int(parent_key)] = replacement
                elif isinstance(container, dict):
                    container[parent_key] = replacement
            else:
                root = replacement
            continue

        if isinstance(current, list):
            stack.append((current, None, None))
            for idx in range(len(current) - 1, -1, -1):
                stack.append((current[idx], (current, None), str(idx)))
            continue

        if isinstance(current, dict):
            stack.append((current, None, None))
            for dict_key, value in reversed(list(current.items())):
                stack.append((value, (current, dict_key), None))
            continue

    return root


def _maybe_resolve_from_cas(obj: Any, cas: ContentAddressedStore | None) -> Any:
    """Resolve CAS markers using iterative traversal."""
    if cas is None:
        return obj

    root: Any = obj
    stack: list[tuple[Any, tuple[Any, ...] | None, str | None]] = [(obj, None, None)]

    while stack:
        current, parent_items, key_in_parent = stack.pop()

        if isinstance(current, list):
            stack.append((current, None, None))
            for idx in range(len(current) - 1, -1, -1):
                stack.append((current[idx], (current, None), str(idx)))
            continue

        if isinstance(current, dict):
            if set(current.keys()) == {"__cas_ref__"} and isinstance(
                current["__cas_ref__"], str
            ):
                replacement = cas.get_text(current["__cas_ref__"])
                if parent_items is not None and key_in_parent is not None:
                    container, parent_key = parent_items, key_in_parent
                    if isinstance(container, list):
                        container[int(parent_key)] = replacement
                    elif isinstance(container, dict):
                        container[parent_key] = replacement
                else:
                    root = replacement
                continue

            stack.append((current, None, None))
            for dict_key, value in reversed(list(current.items())):
                stack.append((value, (current, dict_key), None))
            continue

    return root


def _payload_to_dict(payload: Any) -> dict[str, Any]:
    """Local reimplementation to avoid importing from events at module load."""
    from rfsn_agent.events import _payload_to_dict as events_payload_to_dict

    return events_payload_to_dict(payload)


def _event_from_v1_dict(data: dict[str, Any]) -> HarnessEvent:
    """Convert a v1 serialized event into a v2 HarnessEvent with hashes."""
    from rfsn_agent.events import _payload_from_dict

    header_data = data["header"]
    event_type = str(header_data["event_type"])
    payload = _payload_from_dict(event_type, data["payload"])
    payload_hash = hash_content(canonical_json(_payload_to_dict(payload)))

    header = EventHeader(
        event_id=EventId(header_data["event_id"]),
        trajectory_id=TrajectoryId(header_data["trajectory_id"]),
        sequence=int(header_data["sequence"]),
        event_type=event_type,
        schema_version=CURRENT_EVENT_SCHEMA_VERSION,
        idempotency_key=str(header_data["idempotency_key"]),
        parent_event_id=EventId(header_data["parent_event_id"])
        if header_data.get("parent_event_id") is not None
        else None,
        created_at=datetime.fromisoformat(header_data["created_at"]),
        actor=str(header_data["actor"]),
        action_id=str(header_data["action_id"]),
        previous_event_hash=None,
        event_hash=ContentHash(""),
        previous_signature=None,
        signature=None,
    )
    event_hash = compute_event_hash(header, payload_hash)
    object.__setattr__(header, "event_hash", event_hash)
    return HarnessEvent(header=header, payload=payload, payload_hash=payload_hash)


def _snapshot_from_v1_dict(data: dict[str, Any]) -> HarnessSnapshot:
    """Convert a v1 serialized snapshot into a v2 HarnessSnapshot.

    The old state_hash is ignored and recomputed because the serialized form
    may have used a different schema.
    """
    from rfsn_agent.common import dataclass_from_dict
    from rfsn_agent.domain import (
        BudgetLedger,
        CandidateItem,
        Claim,
        CuratedItem,
        EvidenceLink,
        SubmissionRecord,
        TaskNode,
        ToolInvocation,
        ToolResult,
        VerificationRecord,
    )

    def _list(cls: type[Any], key: str) -> tuple[Any, ...]:
        return tuple(dataclass_from_dict(cls, item) for item in data.get(key, ()))

    # Map verification record results by id so evidence links can derive status.
    record_results: dict[str, VerificationResult] = {
        record["record_id"]: VerificationResult(record["result"])
        for record in data.get("verification_records", ())
    }

    # Rewrite EvidenceLink fields from v1 to v2 before dataclass deserialization.
    links = data.get("evidence_links", ())
    converted_links = []
    for link in links:
        converted = dict(link)
        verification_id = converted.get("verification_id")
        if verification_id is not None and verification_id in record_results:
            result = record_results[verification_id]
            if result == VerificationResult.CONFIRMED:
                converted["current_status"] = VerificationStatus.VERIFIED.value
            elif result == VerificationResult.REFUTED:
                converted["current_status"] = VerificationStatus.REFUTED.value
            else:
                converted["current_status"] = VerificationStatus.INCONCLUSIVE.value
        else:
            verified = converted.pop("verified", False)
            converted["current_status"] = (
                VerificationStatus.VERIFIED.value
                if verified
                else VerificationStatus.UNVERIFIED.value
            )
        converted_links.append(converted)

    budget_data = data.get("budget")
    budget = dataclass_from_dict(BudgetLedger, budget_data) if budget_data is not None else None

    return HarnessSnapshot.create(
        trajectory_id=data["trajectory_id"],
        epoch_id=data["epoch_id"],
        sequence=int(data["sequence"]),
        last_event_hash=None,
        created_at=datetime.fromisoformat(data["created_at"]),
        candidates=_list(CandidateItem, "candidates"),
        curated_items=_list(CuratedItem, "curated_items"),
        claims=_list(Claim, "claims"),
        evidence_links=tuple(
            dataclass_from_dict(EvidenceLink, link) for link in converted_links
        ),
        verification_records=_list(VerificationRecord, "verification_records"),
        tasks=_list(TaskNode, "tasks"),
        budget=budget,
        submissions=_list(SubmissionRecord, "submissions"),
        tool_invocations=_list(ToolInvocation, "tool_invocations"),
        tool_results=_list(ToolResult, "tool_results"),
    )
