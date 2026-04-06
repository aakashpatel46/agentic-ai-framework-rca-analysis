from __future__ import annotations

from abc import ABC, abstractmethod

from validation_enrichment.models import RawTicket


class TicketSource(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def fetch_issue(self, issue_key: str) -> RawTicket:
        raise NotImplementedError

    @abstractmethod
    def download_attachments(self, ticket: RawTicket, destination_dir: str) -> list[str]:
        raise NotImplementedError

