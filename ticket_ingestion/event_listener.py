from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from ticket_ingestion.models import TicketCreatedEvent
from ticket_ingestion.source import EventSource


class CheckpointStore:
    def __init__(self, path: str) -> None:
        self._path = Path(path)

    def load(self) -> datetime | None:
        if not self._path.exists():
            return None

        raw = self._path.read_text(encoding="utf-8").strip()
        if not raw:
            return None

        return datetime.fromisoformat(raw)

    def save(self, value: datetime) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(value.isoformat(), encoding="utf-8")


class EventListener:
    def __init__(
        self,
        source: EventSource,
        on_event: Callable[[TicketCreatedEvent], None],
        checkpoint_store: CheckpointStore,
        poll_interval_seconds: int = 30,
    ) -> None:
        self._source = source
        self._on_event = on_event
        self._checkpoint_store = checkpoint_store
        self._poll_interval_seconds = max(1, poll_interval_seconds)
        self._last_seen = checkpoint_store.load()

    def run_once(self) -> int:
        events = self._source.fetch_new_events(self._last_seen)
        if not events:
            return 0

        latest_seen = self._last_seen
        for event in events:
            self._on_event(event)
            if latest_seen is None or event.created_at > latest_seen:
                latest_seen = event.created_at

        if latest_seen is not None and latest_seen != self._last_seen:
            self._checkpoint_store.save(latest_seen)
            self._last_seen = latest_seen

        return len(events)

    def run_forever(self) -> None:
        while True:
            self.run_once()
            time.sleep(self._poll_interval_seconds)


def build_checkpoint_store_from_env() -> CheckpointStore:
    path = os.getenv("AGENT1_STATE_FILE", ".agent_state/agent1_checkpoint.txt")
    return CheckpointStore(path=path)

