from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class TicketCreatedEvent:
    """Normalized ticket-created event consumed by agents."""

    source: str
    ticket_id: str
    ticket_key: str
    summary: str
    project_key: str
    created_at: datetime
    raw: Any
