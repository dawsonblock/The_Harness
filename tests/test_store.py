"""Tests for SQLite event/snapshot store."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from rfsn_agent.domain import BudgetLedger
from rfsn_agent.events import (
    ActionCommittedPayload,
    HarnessEvent,
    SubmissionRecordedPayload,
    TaskDecomposedPayload,
)
from rfsn_agent.reducer import reduce_event
from rfsn_agent.store import (
    ConcurrentAppendError,
    SQLiteEventStore,
    StoreError,
)


def test_init_trajectory_creates_snapshot() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = SQLiteEventStore(Path(tmp) / "events.db")
        snap = store.init_trajectory(
            "traj-1",
            budget=BudgetLedger(trajectory_id="traj-1", max_tokens=1000),
        )
        assert snap.sequence == 0
        assert snap.trajectory_id == "traj-1"
        assert store.get_event_count("traj-1") == 0


def test_init_duplicate_trajectory_raises() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = SQLiteEventStore(Path(tmp) / "events.db")
        store.init_trajectory("traj-1")
        with pytest.raises(StoreError):
            store.init_trajectory("traj-1")


def test_append_and_replay_single_event() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = SQLiteEventStore(Path(tmp) / "events.db")
        store.init_trajectory(
            "traj-1",
            budget=BudgetLedger(trajectory_id="traj-1", max_tokens=1000),
        )
        event = HarnessEvent.create(
            event_id="evt-1",
            trajectory_id="traj-1",
            sequence=1,
            event_type="action_committed",
            payload=ActionCommittedPayload(action_type="search", action_params=()),
            idempotency_key="idem-1",
        )
        store.append_event(event)
        snap1 = store.get_latest_snapshot("traj-1")
        assert snap1.sequence == 1
        assert snap1.processed_idempotency_keys == {"idem-1"}


def test_replay_produces_same_snapshot_as_in_memory_reduction() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = SQLiteEventStore(Path(tmp) / "events.db")
        snap0 = store.init_trajectory(
            "traj-1",
            budget=BudgetLedger(trajectory_id="traj-1", max_tokens=1000),
        )
        # Use self-contained events that do not depend on prior state.
        events = [
            HarnessEvent.create(
                event_id="evt-1",
                trajectory_id="traj-1",
                sequence=1,
                event_type="action_committed",
                payload=ActionCommittedPayload(action_type="search", action_params=()),
                idempotency_key="idem-1",
            ),
            HarnessEvent.create(
                event_id="evt-2",
                trajectory_id="traj-1",
                sequence=2,
                event_type="task_decomposed",
                payload=TaskDecomposedPayload(
                    parent_task_id=None,
                    task_id="task-1",
                    description="self-contained task",
                    dependency_ids=(),
                ),
                idempotency_key="idem-2",
            ),
        ]
        manual_snap = snap0
        for event in events:
            store.append_event(event)
            manual_snap = reduce_event(manual_snap, event)

        replayed = store.get_latest_snapshot("traj-1")
        assert replayed.state_hash == manual_snap.state_hash


def test_append_duplicate_idempotency_key_ignored_if_identical() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = SQLiteEventStore(Path(tmp) / "events.db")
        store.init_trajectory("traj-1")
        event = HarnessEvent.create(
            event_id="evt-1",
            trajectory_id="traj-1",
            sequence=1,
            event_type="action_committed",
            payload=ActionCommittedPayload(action_type="search", action_params=()),
            idempotency_key="idem-1",
        )
        store.append_event(event)
        store.append_event(event)  # identical replay
        assert store.get_event_count("traj-1") == 1
        snap = store.get_latest_snapshot("traj-1")
        assert snap.sequence == 1


def test_append_conflicting_idempotency_key_raises() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = SQLiteEventStore(Path(tmp) / "events.db")
        store.init_trajectory("traj-1")
        event1 = HarnessEvent.create(
            event_id="evt-1",
            trajectory_id="traj-1",
            sequence=1,
            event_type="action_committed",
            payload=ActionCommittedPayload(action_type="search", action_params=()),
            idempotency_key="idem-1",
        )
        event2 = HarnessEvent.create(
            event_id="evt-2",
            trajectory_id="traj-1",
            sequence=1,
            event_type="action_committed",
            payload=ActionCommittedPayload(action_type="read", action_params=()),
            idempotency_key="idem-1",
        )
        store.append_event(event1)
        with pytest.raises(ConcurrentAppendError):
            store.append_event(event2)


def test_append_out_of_sequence_raises() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = SQLiteEventStore(Path(tmp) / "events.db")
        store.init_trajectory("traj-1")
        event = HarnessEvent.create(
            event_id="evt-5",
            trajectory_id="traj-1",
            sequence=5,
            event_type="action_committed",
            payload=ActionCommittedPayload(action_type="search", action_params=()),
            idempotency_key="idem-5",
        )
        with pytest.raises(ConcurrentAppendError):
            store.append_event(event)


def test_trajectory_isolation() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = SQLiteEventStore(Path(tmp) / "events.db")
        store.init_trajectory("traj-a")
        store.init_trajectory("traj-b")
        event_a = HarnessEvent.create(
            event_id="evt-a1",
            trajectory_id="traj-a",
            sequence=1,
            event_type="action_committed",
            payload=ActionCommittedPayload(action_type="search", action_params=()),
            idempotency_key="idem-a1",
        )
        store.append_event(event_a)
        snap_a = store.get_latest_snapshot("traj-a")
        snap_b = store.get_latest_snapshot("traj-b")
        assert snap_a.sequence == 1
        assert snap_b.sequence == 0
        assert store.list_trajectories() == ["traj-a", "traj-b"]


def test_checkpoint_sequence_tracking() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = SQLiteEventStore(Path(tmp) / "events.db")
        store.init_trajectory("traj-1")
        event = HarnessEvent.create(
            event_id="evt-1",
            trajectory_id="traj-1",
            sequence=1,
            event_type="action_committed",
            payload=ActionCommittedPayload(action_type="search", action_params=()),
            idempotency_key="idem-1",
        )
        store.append_event(event)
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
        event = HarnessEvent.create(
            event_id="evt-1",
            trajectory_id="traj-1",
            sequence=1,
            event_type="submission_recorded",
            payload=SubmissionRecordedPayload(
                submission_id="sub-1", content="answer", source_ids=("src-1",)
            ),
            idempotency_key="idem-1",
        )
        store.append_event(event)
        loaded = store.get_events("traj-1")[0]
        assert loaded.to_dict() == event.to_dict()


def test_replay_from_empty_trajectory_raises() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = SQLiteEventStore(Path(tmp) / "events.db")
        with pytest.raises(StoreError):
            store.replay("missing-traj")
