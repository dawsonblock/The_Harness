"""Tests for SQLite event/snapshot store."""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

from rfsn_agent.domain import BudgetLedger
from rfsn_agent.events import (
    ActionCommittedPayload,
    ProposedEvent,
    SubmissionRecordedPayload,
    TaskDecomposedPayload,
)
from rfsn_agent.reducer import reduce_event
from rfsn_agent.store import (
    IdempotencyConflictError,
    SQLiteEventStore,
    StaleContextError,
    StoreError,
)


def _committed_payload(action_type: str = "search") -> ActionCommittedPayload:
    return ActionCommittedPayload(action_type=action_type, action_params=())


def test_init_trajectory_creates_snapshot() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = SQLiteEventStore(Path(tmp) / "events.db")
        snap = store.init_trajectory(
            "traj-1",
            budget=BudgetLedger(trajectory_id="traj-1", max_tokens=1000),
        )
        assert snap.sequence == 0
        assert snap.trajectory_id == "traj-1"
        assert snap.last_event_hash is None
        assert store.get_event_count("traj-1") == 0


def test_init_duplicate_trajectory_raises() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = SQLiteEventStore(Path(tmp) / "events.db")
        store.init_trajectory("traj-1")
        with pytest.raises(StoreError):
            store.init_trajectory("traj-1")


def test_commit_and_replay_single_event() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = SQLiteEventStore(Path(tmp) / "events.db")
        store.init_trajectory(
            "traj-1",
            budget=BudgetLedger(trajectory_id="traj-1", max_tokens=1000),
        )
        snap0 = store.get_latest_snapshot("traj-1")
        store.commit_events(
            trajectory_id="traj-1",
            expected_sequence=snap0.sequence,
            expected_head_hash=snap0.last_event_hash,
            proposed_events=(
                ProposedEvent(
                    event_type="action_committed",
                    payload=_committed_payload(),
                    idempotency_key="idem-1",
                    actor="policy",
                    action_id="act-1",
                ),
            ),
        )
        snap1 = store.get_latest_snapshot("traj-1")
        assert snap1.sequence == 1
        assert snap1.last_event_hash is not None


def test_replay_produces_same_snapshot_as_in_memory_reduction() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = SQLiteEventStore(Path(tmp) / "events.db")
        snap0 = store.init_trajectory(
            "traj-1",
            budget=BudgetLedger(trajectory_id="traj-1", max_tokens=1000),
        )
        proposed = [
            ProposedEvent(
                event_type="action_committed",
                payload=_committed_payload(),
                idempotency_key="idem-1",
                actor="policy",
                action_id="act-1",
            ),
            ProposedEvent(
                event_type="task_decomposed",
                payload=TaskDecomposedPayload(
                    parent_task_id=None,
                    task_id="task-1",
                    description="self-contained task",
                    dependency_ids=(),
                ),
                idempotency_key="idem-2",
                actor="policy",
                action_id="act-2",
            ),
        ]
        snap = snap0
        for prop in proposed:
            result = store.commit_events(
                trajectory_id="traj-1",
                expected_sequence=snap.sequence,
                expected_head_hash=snap.last_event_hash,
                proposed_events=(prop,),
            )
            snap = reduce_event(
                snap,
                result.committed_events[0],
            )

        replayed = store.get_latest_snapshot("traj-1")
        assert replayed.state_hash == snap.state_hash


def test_commit_duplicate_idempotency_key_ignored_if_identical() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = SQLiteEventStore(Path(tmp) / "events.db")
        store.init_trajectory("traj-1")
        snap0 = store.get_latest_snapshot("traj-1")
        prop = ProposedEvent(
            event_type="action_committed",
            payload=_committed_payload(),
            idempotency_key="idem-1",
            actor="policy",
            action_id="act-1",
        )
        store.commit_events(
            trajectory_id="traj-1",
            expected_sequence=snap0.sequence,
            expected_head_hash=snap0.last_event_hash,
            proposed_events=(prop,),
        )
        # Identical retry should be a no-op.
        store.commit_events(
            trajectory_id="traj-1",
            expected_sequence=1,
            expected_head_hash=store.get_latest_snapshot("traj-1").last_event_hash,
            proposed_events=(prop,),
        )
        assert store.get_event_count("traj-1") == 1
        snap = store.get_latest_snapshot("traj-1")
        assert snap.sequence == 1


def test_commit_conflicting_idempotency_key_raises() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = SQLiteEventStore(Path(tmp) / "events.db")
        store.init_trajectory("traj-1")
        snap0 = store.get_latest_snapshot("traj-1")
        prop1 = ProposedEvent(
            event_type="action_committed",
            payload=_committed_payload("search"),
            idempotency_key="idem-1",
            actor="policy",
            action_id="act-1",
        )
        prop2 = ProposedEvent(
            event_type="action_committed",
            payload=_committed_payload("read"),
            idempotency_key="idem-1",
            actor="policy",
            action_id="act-1",
        )
        store.commit_events(
            trajectory_id="traj-1",
            expected_sequence=snap0.sequence,
            expected_head_hash=snap0.last_event_hash,
            proposed_events=(prop1,),
        )
        with pytest.raises(IdempotencyConflictError):
            store.commit_events(
                trajectory_id="traj-1",
                expected_sequence=1,
                expected_head_hash=store.get_latest_snapshot("traj-1").last_event_hash,
                proposed_events=(prop2,),
            )


def test_commit_out_of_sequence_raises() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = SQLiteEventStore(Path(tmp) / "events.db")
        store.init_trajectory("traj-1")
        snap0 = store.get_latest_snapshot("traj-1")
        prop = ProposedEvent(
            event_type="action_committed",
            payload=_committed_payload(),
            idempotency_key="idem-5",
            actor="policy",
            action_id="act-1",
        )
        with pytest.raises(StaleContextError):
            store.commit_events(
                trajectory_id="traj-1",
                expected_sequence=5,
                expected_head_hash=snap0.last_event_hash,
                proposed_events=(prop,),
            )


def test_trajectory_isolation() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = SQLiteEventStore(Path(tmp) / "events.db")
        store.init_trajectory("traj-a")
        store.init_trajectory("traj-b")
        snap_a0 = store.get_latest_snapshot("traj-a")
        store.commit_events(
            trajectory_id="traj-a",
            expected_sequence=snap_a0.sequence,
            expected_head_hash=snap_a0.last_event_hash,
            proposed_events=(
                ProposedEvent(
                    event_type="action_committed",
                    payload=_committed_payload(),
                    idempotency_key="idem-a1",
                    actor="policy",
                    action_id="act-a1",
                ),
            ),
        )
        snap_a = store.get_latest_snapshot("traj-a")
        snap_b = store.get_latest_snapshot("traj-b")
        assert snap_a.sequence == 1
        assert snap_b.sequence == 0
        assert store.list_trajectories() == ["traj-a", "traj-b"]


def test_checkpoint_sequence_tracking() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = SQLiteEventStore(Path(tmp) / "events.db")
        store.init_trajectory("traj-1")
        snap0 = store.get_latest_snapshot("traj-1")
        store.commit_events(
            trajectory_id="traj-1",
            expected_sequence=snap0.sequence,
            expected_head_hash=snap0.last_event_hash,
            proposed_events=(
                ProposedEvent(
                    event_type="action_committed",
                    payload=_committed_payload(),
                    idempotency_key="idem-1",
                    actor="policy",
                    action_id="act-1",
                ),
            ),
        )
        snap1 = store.get_latest_snapshot("traj-1")
        store.save_snapshot(snap1, event_count=1)
        assert store.get_latest_checkpoint_sequence("traj-1") == 1


def test_wal_mode_enabled() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "events.db"
        with SQLiteEventStore(db_path):
            pass
        with sqlite3.connect(str(db_path)) as conn:
            row = conn.execute("PRAGMA journal_mode").fetchone()
            assert row[0].lower() == "wal"


def test_blob_storage_via_store() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "events.db"
        cas_path = Path(tmp) / "cas"
        store = SQLiteEventStore(db_path, cas_base_dir=cas_path)
        h = store.put_blob("large observation")
        assert store.get_blob(h).decode("utf-8") == "large observation"


def test_event_json_round_trip_through_store() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = SQLiteEventStore(Path(tmp) / "events.db")
        store.init_trajectory("traj-1")
        snap0 = store.get_latest_snapshot("traj-1")
        prop = ProposedEvent(
            event_type="submission_recorded",
            payload=SubmissionRecordedPayload(
                submission_id="sub-1", content="answer", source_ids=("src-1",)
            ),
            idempotency_key="idem-1",
            actor="policy",
            action_id="act-1",
        )
        result = store.commit_events(
            trajectory_id="traj-1",
            expected_sequence=snap0.sequence,
            expected_head_hash=snap0.last_event_hash,
            proposed_events=(prop,),
        )
        loaded = store.get_events("traj-1")[0]
        assert loaded.to_dict() == result.committed_events[0].to_dict()


def test_replay_from_empty_trajectory_raises() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = SQLiteEventStore(Path(tmp) / "events.db")
        with pytest.raises(StoreError):
            store.replay("missing-traj")


def test_commit_invalid_batch_rolls_back_all_events() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = SQLiteEventStore(Path(tmp) / "events.db")
        store.init_trajectory("traj-1")
        snap0 = store.get_latest_snapshot("traj-1")
        # First event is valid; second references an unknown task and will fail.
        props = (
            ProposedEvent(
                event_type="action_committed",
                payload=_committed_payload(),
                idempotency_key="idem-1",
                actor="policy",
                action_id="act-1",
            ),
            ProposedEvent(
                event_type="task_completed",
                payload=TaskDecomposedPayload(
                    parent_task_id=None,
                    task_id="task-1",
                    description="missing",
                    dependency_ids=(),
                ),
                idempotency_key="idem-2",
                actor="policy",
                action_id="act-2",
            ),
        )
        with pytest.raises(StoreError):
            store.commit_events(
                trajectory_id="traj-1",
                expected_sequence=snap0.sequence,
                expected_head_hash=snap0.last_event_hash,
                proposed_events=props,
            )
        snap = store.get_latest_snapshot("traj-1")
        assert snap.sequence == 0
        assert store.get_event_count("traj-1") == 0


def test_commit_events_detects_sql_json_metadata_disagreement() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = SQLiteEventStore(Path(tmp) / "events.db")
        store.init_trajectory("traj-1")
        snap0 = store.get_latest_snapshot("traj-1")
        result = store.commit_events(
            trajectory_id="traj-1",
            expected_sequence=snap0.sequence,
            expected_head_hash=snap0.last_event_hash,
            proposed_events=(
                ProposedEvent(
                    event_type="action_committed",
                    payload=_committed_payload(),
                    idempotency_key="idem-1",
                    actor="policy",
                    action_id="act-1",
                ),
            ),
        )
        event_id = result.committed_events[0].event_id
        # Tamper with the stored JSON so it disagrees with the SQL column.
        conn = sqlite3.connect(str(store.db_path))
        try:
            conn.execute(
                "UPDATE harness_events SET payload_json = ? WHERE event_id = ?",
                ('{"header": {"event_id": "x"}, "payload": {}, "payload_hash": "x"}', event_id),
            )
            conn.commit()
        finally:
            conn.close()
        # Loading events now raises because SQL/JSON disagreement is detected.
        with pytest.raises(StoreError):
            store.get_events("traj-1")


def test_retry_same_command_returns_original_receipt() -> None:
    """A retried logical command should be idempotent despite new physical IDs."""
    with tempfile.TemporaryDirectory() as tmp:
        store = SQLiteEventStore(Path(tmp) / "events.db")
        store.init_trajectory("traj-1")
        snap0 = store.get_latest_snapshot("traj-1")
        prop = ProposedEvent(
            event_type="action_committed",
            payload=_committed_payload(),
            idempotency_key="idem-1",
            actor="policy",
            action_id="act-1",
        )
        result1 = store.commit_events(
            trajectory_id="traj-1",
            expected_sequence=snap0.sequence,
            expected_head_hash=snap0.last_event_hash,
            proposed_events=(prop,),
        )
        snap1 = store.get_latest_snapshot("traj-1")
        result2 = store.commit_events(
            trajectory_id="traj-1",
            expected_sequence=snap1.sequence,
            expected_head_hash=snap1.last_event_hash,
            proposed_events=(prop,),
        )
        assert result1.receipts[0].event_id == result2.receipts[0].event_id
        assert result1.receipts[0].event_hash == result2.receipts[0].event_hash
        assert store.get_event_count("traj-1") == 1


def test_retry_with_same_key_and_same_payload_is_idempotent() -> None:
    """Only the logical command identity matters; store-assigned physical IDs do not
    break retry equivalence.
    """
    with tempfile.TemporaryDirectory() as tmp:
        store = SQLiteEventStore(Path(tmp) / "events.db")
        store.init_trajectory("traj-1")
        snap0 = store.get_latest_snapshot("traj-1")
        prop = ProposedEvent(
            event_type="action_committed",
            payload=_committed_payload(),
            idempotency_key="idem-1",
            actor="policy",
            action_id="act-1",
        )
        result1 = store.commit_events(
            trajectory_id="traj-1",
            expected_sequence=snap0.sequence,
            expected_head_hash=snap0.last_event_hash,
            proposed_events=(prop,),
        )
        snap1 = store.get_latest_snapshot("traj-1")
        result2 = store.commit_events(
            trajectory_id="traj-1",
            expected_sequence=snap1.sequence,
            expected_head_hash=snap1.last_event_hash,
            proposed_events=(prop,),
        )
        # Physical event IDs assigned by the store differ between calls, but the
        # idempotency receipt points back to the original committed event.
        assert result1.receipts[0].event_id == result2.receipts[0].event_id
        assert store.get_event_count("traj-1") == 1


def test_v1_to_v2_migration_preserves_replay() -> None:
    """A database created with the v1 schema can be opened and replayed under v2."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "events.db"

        # Create a v1 schema manually.
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute(
                """
                CREATE TABLE schema_version (
                    version INTEGER PRIMARY KEY
                )
                """
            )
            conn.execute("INSERT INTO schema_version (version) VALUES (1)")
            conn.execute(
                """
                CREATE TABLE trajectories (
                    trajectory_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    epoch_id TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE harness_events (
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
                    UNIQUE(trajectory_id, sequence),
                    UNIQUE(trajectory_id, idempotency_key)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE harness_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    trajectory_id TEXT NOT NULL,
                    epoch_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    state_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    event_count INTEGER NOT NULL,
                    snapshot_json TEXT NOT NULL,
                    UNIQUE(trajectory_id, sequence)
                )
                """
            )

            # Insert a v1 trajectory and snapshot.
            conn.execute(
                "INSERT INTO trajectories (trajectory_id, created_at, epoch_id) VALUES (?, ?, ?)",
                ("traj-1", "2024-01-01T00:00:00+00:00", "epoch-0"),
            )
            v1_snapshot = {
                "trajectory_id": "traj-1",
                "epoch_id": "epoch-0",
                "sequence": 0,
                "state_hash": "a" * 64,
                "created_at": "2024-01-01T00:00:00+00:00",
                "candidates": [],
                "curated_items": [],
                "claims": [],
                "evidence_links": [],
                "verification_records": [],
                "tasks": [],
                "budget": {
                    "trajectory_id": "traj-1",
                    "max_tokens": 1000,
                    "tokens_used": 0,
                    "tokens_reserved": 0,
                    "max_tool_calls": None,
                    "tool_calls_used": 0,
                    "max_wall_seconds": None,
                    "wall_seconds_used": 0.0,
                },
                "submissions": [],
                "tool_invocations": [],
                "tool_results": [],
                "processed_idempotency_keys": [],
            }
            conn.execute(
                """
                INSERT INTO harness_snapshots (
                    snapshot_id, trajectory_id, epoch_id, sequence, state_hash,
                    created_at, event_count, snapshot_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "snap-0",
                    "traj-1",
                    "epoch-0",
                    0,
                    "a" * 64,
                    "2024-01-01T00:00:00+00:00",
                    0,
                    json.dumps(v1_snapshot, sort_keys=True),
                ),
            )

            # Insert a v1 event.
            v1_event = {
                "header": {
                    "event_id": "evt-1",
                    "trajectory_id": "traj-1",
                    "sequence": 1,
                    "event_type": "action_committed",
                    "schema_version": 1,
                    "idempotency_key": "idem-1",
                    "parent_event_id": None,
                    "created_at": "2024-01-01T00:00:01+00:00",
                    "actor": "policy",
                    "action_id": "act-1",
                },
                "payload": {"action_type": "search", "action_params": []},
                "payload_hash": "b" * 64,
            }
            conn.execute(
                """
                INSERT INTO harness_events (
                    event_id, trajectory_id, sequence, event_type, schema_version,
                    idempotency_key, parent_event_id, created_at, actor, action_id,
                    payload_hash, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "evt-1",
                    "traj-1",
                    1,
                    "action_committed",
                    1,
                    "idem-1",
                    None,
                    "2024-01-01T00:00:01+00:00",
                    "policy",
                    "act-1",
                    "b" * 64,
                    json.dumps(v1_event, sort_keys=True),
                ),
            )
            conn.commit()
        finally:
            conn.close()

        # Opening the store should migrate v1 -> v2.
        store = SQLiteEventStore(db_path)
        snap = store.get_latest_snapshot("traj-1")
        assert snap.sequence == 1
        assert snap.last_event_hash is not None
        assert snap.last_event_hash != ""
        events = store.get_events("traj-1")
        assert len(events) == 1
        assert events[0].header.previous_event_hash is None
        assert events[0].header.event_hash is not None
        assert events[0].header.event_hash != ""


def _signed_store(tmp: str, key: bytes) -> SQLiteEventStore:
    return SQLiteEventStore(Path(tmp) / "events.db", signing_key=key)


def test_commit_with_signing_key_signs_events() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        key = b"signing-key"
        store = _signed_store(tmp, key)
        store.init_trajectory(
            "traj-1",
            budget=BudgetLedger(trajectory_id="traj-1", max_tokens=1000),
        )
        snap0 = store.get_latest_snapshot("traj-1")
        result = store.commit_events(
            trajectory_id="traj-1",
            expected_sequence=snap0.sequence,
            expected_head_hash=snap0.last_event_hash,
            proposed_events=(
                ProposedEvent(
                    event_type="action_committed",
                    payload=_committed_payload(),
                    idempotency_key="idem-1",
                    actor="policy",
                    action_id="act-1",
                ),
            ),
        )
        event = result.committed_events[0]
        assert event.header.signature is not None
        assert event.header.previous_signature is None

        events = store.get_events("traj-1")
        assert events[0].header.signature == event.header.signature


def test_signed_chain_requires_consistent_signatures() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        key = b"signing-key"
        store = _signed_store(tmp, key)
        store.init_trajectory(
            "traj-1",
            budget=BudgetLedger(trajectory_id="traj-1", max_tokens=1000),
        )
        snap0 = store.get_latest_snapshot("traj-1")
        store.commit_events(
            trajectory_id="traj-1",
            expected_sequence=snap0.sequence,
            expected_head_hash=snap0.last_event_hash,
            proposed_events=(
                ProposedEvent(
                    event_type="action_committed",
                    payload=_committed_payload(),
                    idempotency_key="idem-1",
                    actor="policy",
                    action_id="act-1",
                ),
            ),
        )

        # Tamper with the stored signature of the first event.
        db_path = Path(tmp) / "events.db"
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute(
                "UPDATE harness_events SET signature = ? WHERE sequence = 1",
                ("a" * 64,),
            )
            conn.commit()
        finally:
            conn.close()

        # Replaying should detect the signature mismatch.
        tampered_store = _signed_store(tmp, key)
        with pytest.raises(StoreError):
            tampered_store.get_latest_snapshot("traj-1")


def test_signed_store_rejects_unsigned_event_in_chain() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        # Create an unsigned store and commit an event.
        store = SQLiteEventStore(Path(tmp) / "events.db")
        store.init_trajectory(
            "traj-1",
            budget=BudgetLedger(trajectory_id="traj-1", max_tokens=1000),
        )
        snap0 = store.get_latest_snapshot("traj-1")
        store.commit_events(
            trajectory_id="traj-1",
            expected_sequence=snap0.sequence,
            expected_head_hash=snap0.last_event_hash,
            proposed_events=(
                ProposedEvent(
                    event_type="action_committed",
                    payload=_committed_payload(),
                    idempotency_key="idem-1",
                    actor="policy",
                    action_id="act-1",
                ),
            ),
        )
        store.close()

        # Reopening an already-migrated unsigned store with a signing key
        # should fail because existing events are not signed.
        key = b"signing-key"
        signed_store = SQLiteEventStore(Path(tmp) / "events.db", signing_key=key)
        with pytest.raises(StoreError):
            signed_store.get_events("traj-1")


def test_v1_to_v3_migration_with_signing_key() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "events.db"
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute(
                """
                CREATE TABLE schema_version (
                    version INTEGER PRIMARY KEY
                )
                """
            )
            conn.execute("INSERT INTO schema_version (version) VALUES (1)")
            conn.execute(
                """
                CREATE TABLE trajectories (
                    trajectory_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    epoch_id TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                INSERT INTO trajectories (trajectory_id, created_at, epoch_id)
                VALUES (?, ?, ?)
                """,
                ("traj-1", "2024-01-01T00:00:00+00:00", "epoch-0"),
            )
            conn.execute(
                """
                CREATE TABLE harness_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    trajectory_id TEXT NOT NULL,
                    epoch_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    state_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    event_count INTEGER NOT NULL,
                    snapshot_json TEXT NOT NULL
                )
                """
            )
            from rfsn_agent.domain import HarnessSnapshot

            snap = HarnessSnapshot.create(
                trajectory_id="traj-1",
                epoch_id="epoch-0",
                sequence=0,
            )
            conn.execute(
                """
                INSERT INTO harness_snapshots (
                    snapshot_id, trajectory_id, epoch_id, sequence, state_hash,
                    created_at, event_count, snapshot_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "snap-0",
                    "traj-1",
                    "epoch-0",
                    0,
                    snap.state_hash,
                    snap.created_at.isoformat(),
                    0,
                    json.dumps(snap.to_dict(), sort_keys=True),
                ),
            )
            conn.execute(
                """
                CREATE TABLE harness_events (
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
                    payload_json TEXT NOT NULL
                )
                """
            )
            v1_event = {
                "header": {
                    "event_id": "evt-1",
                    "trajectory_id": "traj-1",
                    "sequence": 1,
                    "event_type": "action_committed",
                    "schema_version": 1,
                    "idempotency_key": "idem-1",
                    "parent_event_id": None,
                    "created_at": "2024-01-01T00:00:01+00:00",
                    "actor": "policy",
                    "action_id": "act-1",
                },
                "payload": {"action_type": "search", "action_params": []},
                "payload_hash": "b" * 64,
            }
            conn.execute(
                """
                INSERT INTO harness_events (
                    event_id, trajectory_id, sequence, event_type, schema_version,
                    idempotency_key, parent_event_id, created_at, actor, action_id,
                    payload_hash, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "evt-1",
                    "traj-1",
                    1,
                    "action_committed",
                    1,
                    "idem-1",
                    None,
                    "2024-01-01T00:00:01+00:00",
                    "policy",
                    "act-1",
                    "b" * 64,
                    json.dumps(v1_event, sort_keys=True),
                ),
            )
            conn.commit()
        finally:
            conn.close()

        key = b"migration-key"
        store = SQLiteEventStore(db_path, signing_key=key)
        snap = store.get_latest_snapshot("traj-1")
        assert snap.sequence == 1
        assert snap.last_signature is not None
        events = store.get_events("traj-1")
        assert len(events) == 1
        assert events[0].header.signature is not None


def test_automatic_checkpoint_at_sequence_50(monkeypatch: pytest.MonkeyPatch) -> None:
    import rfsn_agent.reducer
    import rfsn_agent.store
    with tempfile.TemporaryDirectory() as tmp:
        store = SQLiteEventStore(Path(tmp) / "events.db", cas_base_dir=Path(tmp) / "cas")
        store.init_trajectory("traj-1")
        snap = store.get_latest_snapshot("traj-1")

        reduce_calls: list[int] = []
        original_reduce = rfsn_agent.reducer.reduce_event

        def counting_reduce(snap, event):
            reduce_calls.append(event.sequence)
            return original_reduce(snap, event)

        monkeypatch.setattr(rfsn_agent.store, "reduce_event", counting_reduce)

        expected_sequence = snap.sequence
        expected_head_hash = snap.last_event_hash
        for i in range(50):
            prop = ProposedEvent(
                event_type="action_committed",
                payload=_committed_payload(),
                idempotency_key=f"idem-{i}",
                actor="policy",
                action_id=f"act-{i}",
            )
            result = store.commit_events(
                trajectory_id="traj-1",
                expected_sequence=expected_sequence,
                expected_head_hash=expected_head_hash,
                proposed_events=(prop,),
            )
            expected_sequence = result.last_sequence
            expected_head_hash = result.head_hash

        assert store.get_latest_checkpoint_sequence("traj-1") == 50

        # Verify loading the head uses the checkpoint (no replay needed)
        reduce_calls.clear()
        head = store.get_latest_snapshot("traj-1")
        assert head.sequence == 50
        assert len(reduce_calls) == 0


def test_cas_offloading_large_payload() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cas_dir = Path(tmp) / "cas"
        store = SQLiteEventStore(Path(tmp) / "events.db", cas_base_dir=cas_dir)
        store.init_trajectory("traj-1")
        snap0 = store.get_latest_snapshot("traj-1")
        large_content = "A" * 5000
        prop = ProposedEvent(
            event_type="submission_recorded",
            payload=SubmissionRecordedPayload(
                submission_id="sub-1",
                content=large_content,
                source_ids=("src-1",),
            ),
            idempotency_key="idem-large",
            actor="policy",
            action_id="act-1",
        )
        store.commit_events(
            trajectory_id="traj-1",
            expected_sequence=snap0.sequence,
            expected_head_hash=snap0.last_event_hash,
            proposed_events=(prop,),
        )
        # Verify the event can be loaded and content is intact
        events = store.get_events("traj-1")
        assert len(events) == 1
        payload = events[0].payload
        assert isinstance(payload, SubmissionRecordedPayload)
        assert payload.content == large_content

        # Verify the payload_json in DB contains the CAS reference, not the raw string
        conn = sqlite3.connect(str(store.db_path))
        cur = conn.cursor()
        row = cur.execute(
            "SELECT payload_json FROM harness_events WHERE trajectory_id = ?",
            ("traj-1",),
        ).fetchone()
        assert row is not None
        data = json.loads(row[0])
        cas_ref = data["payload"]["content"]
        assert isinstance(cas_ref, dict) and "__cas_ref__" in cas_ref
        # Verify the hash exists in CAS
        h = cas_ref["__cas_ref__"]
        assert store.cas is not None
        assert store.cas.exists(h) is True
        assert store.cas.get_text(h) == large_content
        conn.close()
