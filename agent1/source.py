from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from agent1.models import TicketCreatedEvent


class EventSource(ABC):
    """Base contract for any external event source."""

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def fetch_new_events(self, last_seen: datetime | None) -> list[TicketCreatedEvent]:
        """Return events strictly newer than `last_seen` in ascending created order."""
        raise NotImplementedError
