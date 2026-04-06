from __future__ import annotations

import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from agent1.event_listener import CheckpointStore, EventListener
from agent1.models import TicketCreatedEvent
from agent1.source import EventSource


class FakeSource(EventSource):
    def __init__(self, responses: list[list[TicketCreatedEvent]]) -> None:
        self._responses = responses
        self._calls = 0
        self.last_seen_args: list[datetime | None] = []

    @property
    def name(self) -> str:
        return "fake"

    def fetch_new_events(self, last_seen: datetime | None) -> list[TicketCreatedEvent]:
        self.last_seen_args.append(last_seen)
        response = self._responses[self._calls] if self._calls < len(self._responses) else []
        self._calls += 1
        return response


def _event(key: str, created: str) -> TicketCreatedEvent:
    return TicketCreatedEvent(
        source="fake",
        ticket_id=key,
        ticket_key=key,
        summary="s",
        project_key="P",
        created_at=datetime.fromisoformat(created),
        raw=None,
    )


class EventListenerTests(unittest.TestCase):
    def test_listener_processes_events_and_saves_checkpoint(self) -> None:
        with TemporaryDirectory() as td:
            store = CheckpointStore(str(Path(td) / "checkpoint.txt"))
            e1 = _event("P-1", "2026-03-11T10:00:00+00:00")
            e2 = _event("P-2", "2026-03-11T10:05:00+00:00")
            source = FakeSource([[e1, e2]])
            received: list[str] = []

            listener = EventListener(
                source=source,
                on_event=lambda ev: received.append(ev.ticket_key),
                checkpoint_store=store,
                poll_interval_seconds=1,
            )

            processed = listener.run_once()

            self.assertEqual(processed, 2)
            self.assertEqual(received, ["P-1", "P-2"])
            self.assertEqual(
                store.load(),
                datetime(2026, 3, 11, 10, 5, tzinfo=timezone.utc),
            )

    def test_listener_uses_existing_checkpoint(self) -> None:
        with TemporaryDirectory() as td:
            store = CheckpointStore(str(Path(td) / "checkpoint.txt"))
            prior = datetime(2026, 3, 11, 9, 0, tzinfo=timezone.utc)
            store.save(prior)

            source = FakeSource([[]])
            listener = EventListener(
                source=source,
                on_event=lambda _: None,
                checkpoint_store=store,
                poll_interval_seconds=1,
            )

            processed = listener.run_once()

            self.assertEqual(processed, 0)
            self.assertEqual(source.last_seen_args, [prior])


if __name__ == "__main__":
    unittest.main()
