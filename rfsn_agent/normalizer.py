"""Normalize tool results into candidate events automatically."""

from __future__ import annotations

from rfsn_agent.events import (
    CandidateAddedPayload,
    ProposedEvent,
    ToolResultReceivedPayload,
)
from rfsn_agent.store import SQLiteEventStore
from rfsn_agent.types import ItemId, TrajectoryId


class ResultNormalizer:
    """Scans tool results and auto-creates CandidateAdded events for successes."""

    def __init__(self, store: SQLiteEventStore) -> None:
        self.store = store

    def normalize(self, trajectory_id: str) -> None:
        """Scan ``trajectory_id`` for successful tool results and commit candidates.

        Each successful ``tool_result_received`` event that does not already have
        a corresponding auto-generated candidate produces a new
        ``candidate_added`` event committed atomically via the store.
        """
        snapshot = self.store.get_latest_snapshot(trajectory_id)
        events = self.store.get_events(trajectory_id)

        existing_auto_candidates = {c.item_id for c in snapshot.candidates}
        proposed: list[ProposedEvent] = []

        for event in events:
            if event.event_type != "tool_result_received":
                continue
            payload = event.payload
            if not isinstance(payload, ToolResultReceivedPayload):
                continue
            if payload.status != "success":
                continue

            item_id = ItemId(f"auto-cand-{payload.invocation_id}")
            if item_id in existing_auto_candidates:
                continue

            proposed.append(
                ProposedEvent(
                    event_type="candidate_added",
                    payload=CandidateAddedPayload(
                        item_id=item_id,
                        trajectory_id=TrajectoryId(trajectory_id),
                        source_id=str(payload.invocation_id),
                        retrieval_query="",
                        content=payload.content,
                    ),
                    idempotency_key=f"auto-cand-{payload.invocation_id}",
                    actor="normalizer",
                    action_id="normalize",
                )
            )

        if not proposed:
            return

        self.store.commit_events(
            trajectory_id=TrajectoryId(trajectory_id),
            expected_sequence=snapshot.sequence,
            expected_head_hash=snapshot.last_event_hash,
            proposed_events=tuple(proposed),
        )
