from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime

from jira import JIRA
from jira.exceptions import JIRAError

from ticket_ingestion.models import TicketCreatedEvent
from ticket_ingestion.source import EventSource


@dataclass(frozen=True)
class JiraConfig:
    base_url: str
    email: str
    api_token: str
    project_key: str
    jql_extra: str = ""
    page_size: int = 100


class JiraSource(EventSource):
    CREATED_FORMATS = ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z")

    def __init__(self, config: JiraConfig, client: JIRA | None = None) -> None:
        self._config = config
        self._client = client or JIRA(
            server=config.base_url,
            basic_auth=(config.email, config.api_token),
        )

    @property
    def name(self) -> str:
        return "jira"

    @classmethod
    def from_env(cls) -> "JiraSource":
        base_url = os.getenv("JIRA_BASE_URL") or os.getenv("JIRA_SERVER")
        required = {
            "JIRA_BASE_URL": base_url,
            "JIRA_EMAIL": os.getenv("JIRA_EMAIL"),
            "JIRA_API_TOKEN": os.getenv("JIRA_API_TOKEN"),
            "JIRA_PROJECT_KEY": os.getenv("JIRA_PROJECT_KEY"),
        }
        missing = [key for key, value in required.items() if not value]
        if missing:
            raise ValueError(f"Missing required Jira env vars: {', '.join(missing)}")

        config = JiraConfig(
            base_url=base_url or "",
            email=required["JIRA_EMAIL"] or "",
            api_token=required["JIRA_API_TOKEN"] or "",
            project_key=required["JIRA_PROJECT_KEY"] or "",
            jql_extra=os.getenv("JIRA_JQL_EXTRA", "").strip(),
            page_size=int(os.getenv("JIRA_PAGE_SIZE", "100")),
        )
        return cls(config=config)

    def fetch_new_events(self, last_seen: datetime | None) -> list[TicketCreatedEvent]:
        jql = self._build_jql(last_seen)
        issues = self._search_all_issues(jql)
        events = [self._to_event(issue) for issue in issues]

        if last_seen is None:
            return events

        return [event for event in events if event.created_at > last_seen]

    def fetch_ticket_by_key(self, issue_key: str) -> TicketCreatedEvent:
        issue = self._client.issue(issue_key, fields="summary,created,project")
        return self._to_event(issue)

    def _search_all_issues(self, jql: str) -> list:
        fields = "summary,created,project"
        # Jira Cloud deprecated /search for some tenants. Prefer enhanced_search_issues.
        if hasattr(self._client, "enhanced_search_issues"):
            try:
                return list(
                    self._client.enhanced_search_issues(
                        jql,
                        maxResults=False,
                        fields=fields,
                    )
                )
            except JIRAError:
                # Fall back to legacy search path for compatibility.
                pass

        start_at = 0
        page_size = max(1, self._config.page_size)
        issues: list = []

        while True:
            try:
                batch = self._client.search_issues(
                    jql,
                    startAt=start_at,
                    maxResults=page_size,
                    fields=fields,
                )
            except JIRAError as exc:
                # If legacy endpoint is explicitly blocked, retry once with enhanced API.
                msg = str(exc).lower()
                if "deprecated" in msg and "enhanced_search_issues" in msg and hasattr(
                    self._client, "enhanced_search_issues"
                ):
                    return list(
                        self._client.enhanced_search_issues(
                            jql,
                            maxResults=False,
                            fields=fields,
                        )
                    )
                raise
            if not batch:
                break

            issues.extend(batch)
            if len(batch) < page_size:
                break
            start_at += page_size

        return issues

    def _build_jql(self, last_seen: datetime | None) -> str:
        clauses = [f'project = "{self._config.project_key}"']
        if last_seen is not None:
            timestamp = last_seen.astimezone().strftime("%Y-%m-%d %H:%M")
            clauses.append(f'created >= "{timestamp}"')
        if self._config.jql_extra:
            clauses.append(f"({self._config.jql_extra})")

        return " AND ".join(clauses) + " ORDER BY created ASC"

    def _to_event(self, issue) -> TicketCreatedEvent:
        created_at = self._parse_created(issue.fields.created)
        project_key = getattr(getattr(issue.fields, "project", None), "key", self._config.project_key)
        return TicketCreatedEvent(
            source=self.name,
            ticket_id=str(issue.id),
            ticket_key=str(issue.key),
            summary=getattr(issue.fields, "summary", "") or "",
            project_key=project_key,
            created_at=created_at,
            raw=issue,
        )

    @classmethod
    def _parse_created(cls, value: str) -> datetime:
        for fmt in cls.CREATED_FORMATS:
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
        raise ValueError(f"Unsupported Jira created timestamp format: {value}")

