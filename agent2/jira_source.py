from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from jira import JIRA

from agent2.models import AttachmentMeta, RawTicket
from agent2.source import TicketSource


@dataclass(frozen=True)
class JiraSourceConfig:
    base_url: str
    email: str
    api_token: str
    steps_field_id: str
    affects_version_field_id: str
    organizations_field_id: str


class JiraTicketSource(TicketSource):
    TIMESTAMP_FORMATS = ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z")
    COMMENT_TIMESTAMP_FORMATS = ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z")

    def __init__(self, config: JiraSourceConfig, client: JIRA | None = None) -> None:
        self._config = config
        self._client = client or JIRA(
            server=config.base_url,
            basic_auth=(config.email, config.api_token),
        )
        self.last_attachment_stats: dict[str, int] = {"downloaded_count": 0, "reused_count": 0}

    @property
    def name(self) -> str:
        return "jira"

    @classmethod
    def from_env(cls) -> "JiraTicketSource":
        base_url = os.getenv("JIRA_BASE_URL") or os.getenv("JIRA_SERVER")
        required = {
            "JIRA_BASE_URL": base_url,
            "JIRA_EMAIL": os.getenv("JIRA_EMAIL"),
            "JIRA_API_TOKEN": os.getenv("JIRA_API_TOKEN"),
        }
        missing = [key for key, value in required.items() if not value]
        if missing:
            raise ValueError(f"Missing required Jira env vars: {', '.join(missing)}")

        return cls(
            config=JiraSourceConfig(
                base_url=base_url or "",
                email=required["JIRA_EMAIL"] or "",
                api_token=required["JIRA_API_TOKEN"] or "",
                steps_field_id=(os.getenv("JIRA_STEPS_TO_REPRO_FIELD") or "").strip(),
                affects_version_field_id=(os.getenv("JIRA_AFFECTS_VERSION_FIELD") or "").strip(),
                organizations_field_id=(
                    os.getenv("JIRA_ORGANIZATION_FIELD")
                    or os.getenv("JIRA_ORGANIZATIONS_FIELD")
                    or "customfield_10002"
                ).strip(),
            )
        )

    def fetch_issue(self, issue_key: str) -> RawTicket:
        base_fields = [
            "summary",
            "description",
            "issuetype",
            "status",
            "priority",
            "reporter",
            "assignee",
            "labels",
            "comment",
            "created",
            "updated",
            "attachment",
        ]
        if self._config.steps_field_id:
            base_fields.append(self._config.steps_field_id)
        if self._config.affects_version_field_id:
            base_fields.append(self._config.affects_version_field_id)
        if self._config.organizations_field_id:
            base_fields.append(self._config.organizations_field_id)
        issue = self._client.issue(
            issue_key,
            fields=",".join(base_fields),
        )

        attachments = [
            AttachmentMeta(
                id=str(att.id),
                filename=str(att.filename),
                mime_type=getattr(att, "mimeType", None),
                size=getattr(att, "size", None),
                content_url=str(att.content),
            )
            for att in getattr(issue.fields, "attachment", [])
        ]

        return RawTicket(
            source=self.name,
            issue_id=str(issue.id),
            issue_key=str(issue.key),
            summary=getattr(issue.fields, "summary", "") or "",
            description=self._description_to_text(getattr(issue.fields, "description", "")),
            steps_to_reproduce=self._extract_steps_to_reproduce(issue),
            affects_version=self._extract_affects_version(issue),
            issue_type=getattr(getattr(issue.fields, "issuetype", None), "name", "") or "",
            status=getattr(getattr(issue.fields, "status", None), "name", "") or "",
            priority=getattr(getattr(issue.fields, "priority", None), "name", "") or "",
            reporter=getattr(getattr(issue.fields, "reporter", None), "displayName", "") or "",
            assignee=getattr(getattr(issue.fields, "assignee", None), "displayName", "") or "",
            labels=list(getattr(issue.fields, "labels", []) or []),
            organizations=self._extract_organizations(issue),
            comments=self._extract_comments(issue),
            created_at=self._parse_timestamp(getattr(issue.fields, "created", "")),
            updated_at=self._parse_timestamp(getattr(issue.fields, "updated", "")),
            attachments=attachments,
            raw=issue.raw,
        )

    def download_attachments(self, ticket: RawTicket, destination_dir: str) -> list[str]:
        output_dir = Path(destination_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths: list[str] = []
        downloaded_count = 0
        reused_count = 0
        for attachment in ticket.attachments:
            target = output_dir / attachment.filename
            if target.exists():
                existing_size = target.stat().st_size
                remote_size = attachment.size
                if remote_size is None or int(remote_size) == int(existing_size):
                    reused_count += 1
                    saved_paths.append(str(target.resolve()))
                    continue

            response = self._client._session.get(attachment.content_url, stream=True)
            response.raise_for_status()
            with target.open("wb") as file_handle:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file_handle.write(chunk)
            downloaded_count += 1
            saved_paths.append(str(target.resolve()))

        self.last_attachment_stats = {
            "downloaded_count": downloaded_count,
            "reused_count": reused_count,
        }
        return saved_paths

    @classmethod
    def _parse_timestamp(cls, value: str) -> datetime:
        for fmt in cls.TIMESTAMP_FORMATS:
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
        raise ValueError(f"Unsupported Jira timestamp format: {value}")

    @classmethod
    def _description_to_text(cls, description: object) -> str:
        if isinstance(description, str):
            return description
        if description is None:
            return ""
        return str(description)

    def _extract_steps_to_reproduce(self, issue) -> str:
        if not self._config.steps_field_id:
            return ""
        value = getattr(issue.fields, self._config.steps_field_id, "")
        return self._description_to_text(value).strip()

    def _extract_affects_version(self, issue) -> str:
        if not self._config.affects_version_field_id:
            return ""
        value = getattr(issue.fields, self._config.affects_version_field_id, "")
        if isinstance(value, list):
            pieces: list[str] = []
            for item in value:
                name = getattr(item, "name", None)
                pieces.append(str(name or item).strip())
            return ", ".join([piece for piece in pieces if piece])
        name = getattr(value, "name", None)
        return self._description_to_text(name or value).strip()

    def _extract_organizations(self, issue) -> list[str]:
        out: list[str] = []
        if self._config.organizations_field_id:
            raw_value = getattr(issue.fields, self._config.organizations_field_id, None)
            out.extend(self._normalize_organizations(raw_value))
        if out:
            return out

        # Heuristic fallback when organizations custom field id is not configured.
        raw_fields = issue.raw.get("fields", {}) if isinstance(issue.raw, dict) else {}
        if isinstance(raw_fields, dict):
            for key, value in raw_fields.items():
                if "organization" in str(key).lower():
                    out.extend(self._normalize_organizations(value))
        deduped: list[str] = []
        seen: set[str] = set()
        for item in out:
            v = item.strip()
            if v and v.lower() not in seen:
                seen.add(v.lower())
                deduped.append(v)
        return deduped

    def _normalize_organizations(self, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            out: list[str] = []
            for item in value:
                name = getattr(item, "name", None) if not isinstance(item, dict) else item.get("name")
                out.append(str(name or item).strip())
            return [x for x in out if x]
        if isinstance(value, dict):
            name = value.get("name") or value.get("value")
            return [str(name).strip()] if str(name or "").strip() else []
        name = getattr(value, "name", None)
        text = str(name or value).strip()
        return [text] if text else []

    def _extract_comments(self, issue) -> list[str]:
        comment_block = getattr(issue.fields, "comment", None)
        comment_items = getattr(comment_block, "comments", None)
        if not isinstance(comment_items, list):
            return []
        out: list[str] = []
        for row in comment_items:
            body = getattr(row, "body", "")
            text = self._description_to_text(body).strip()
            if text:
                author_obj = getattr(row, "author", None)
                author_name = (
                    getattr(author_obj, "displayName", None)
                    or getattr(author_obj, "name", None)
                    or getattr(author_obj, "accountId", None)
                    or "Unknown"
                )
                created_raw = str(getattr(row, "created", "") or "").strip()
                created_friendly = self._format_comment_time(created_raw)
                out.append(f"{author_name} ({created_friendly}): {text}")
        return out

    def _format_comment_time(self, value: str) -> str:
        if not value:
            return "Unknown time"
        for fmt in self.COMMENT_TIMESTAMP_FORMATS:
            try:
                dt = datetime.strptime(value, fmt)
                return dt.strftime("%d/%b/%y %I:%M %p")
            except ValueError:
                continue
        return value
