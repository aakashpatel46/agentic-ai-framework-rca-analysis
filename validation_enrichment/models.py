from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class AttachmentMeta:
    id: str
    filename: str
    mime_type: str | None
    size: int | None
    content_url: str


@dataclass(frozen=True)
class RawTicket:
    source: str
    issue_id: str
    issue_key: str
    summary: str
    description: str
    steps_to_reproduce: str
    affects_version: str
    issue_type: str
    status: str
    priority: str
    reporter: str
    assignee: str
    labels: list[str]
    organizations: list[str]
    comments: list[str]
    created_at: datetime
    updated_at: datetime
    attachments: list[AttachmentMeta]
    raw: Any


@dataclass(frozen=True)
class ValidationEnrichmentResult:
    source: str
    issue_key: str
    category: str
    required_information: list[str]
    available_information: list[str]
    missing_information: list[str]
    attachment_paths: list[str]
    secondary_openai_called: bool
    secondary_enrichment: dict[str, Any]
    raw_ticket: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["raw_ticket"]["created_at"] = self.raw_ticket["created_at"].isoformat()
        payload["raw_ticket"]["updated_at"] = self.raw_ticket["updated_at"].isoformat()
        return payload
