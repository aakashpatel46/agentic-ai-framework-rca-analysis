from __future__ import annotations

from datetime import datetime, timezone

from ticket_ingestion.agent1 import handle_ticket_created
from ticket_ingestion.event_listener import CheckpointStore, EventListener
from ticket_ingestion.models import TicketCreatedEvent
from ticket_ingestion.source import EventSource


class DummySource(EventSource):
    """Sends one fake ticket-created event on first poll."""

    def __init__(self) -> None:
        self._sent = False

    @property
    def name(self) -> str:
        return "dummy"

    def fetch_new_events(self, last_seen: datetime | None) -> list[TicketCreatedEvent]:
        if self._sent:
            return []

        self._sent = True
        return [
            TicketCreatedEvent(
                source=self.name,
                ticket_id="10001",
                ticket_key="DUMMY-1",
                summary="Dummy ticket for manual testing",
                project_key="DUMMY",
                created_at=datetime.now(timezone.utc),
                raw={"manual_test": True},
            )
        ]


def main() -> None:
    listener = EventListener(
        source=DummySource(),
        on_event=handle_ticket_created,
        checkpoint_store=CheckpointStore(".agent_state/agent1_dummy_checkpoint.txt"),
        poll_interval_seconds=1,
    )

    processed = listener.run_once()
    print(f"[dummy_tests] Processed events: {processed}")


if __name__ == "__main__":
    main()

