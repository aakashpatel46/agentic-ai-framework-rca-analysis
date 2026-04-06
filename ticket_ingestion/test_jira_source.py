from __future__ import annotations

import unittest
from types import SimpleNamespace

from ticket_ingestion.jira_source import JiraConfig, JiraSource


class FakeIssue:
    def __init__(self, issue_id: str, key: str, created: str, summary: str, project_key: str = "DEMO") -> None:
        self.id = issue_id
        self.key = key
        self.fields = SimpleNamespace(
            created=created,
            summary=summary,
            project=SimpleNamespace(key=project_key),
        )


class FakeClient:
    def __init__(self, responses: list[list[FakeIssue]]) -> None:
        self._responses = responses
        self.calls: list[dict[str, object]] = []

    def search_issues(self, jql: str, startAt: int, maxResults: int, fields: str):
        self.calls.append(
            {
                "jql": jql,
                "startAt": startAt,
                "maxResults": maxResults,
                "fields": fields,
            }
        )
        index = startAt // maxResults
        return self._responses[index] if index < len(self._responses) else []


class JiraSourceTests(unittest.TestCase):
    def test_jira_source_builds_project_jql_and_maps_event(self) -> None:
        cfg = JiraConfig(
            server="https://example.atlassian.net",
            email="a@b.com",
            api_token="token",
            project_key="ABC",
            page_size=2,
        )

        issue1 = FakeIssue("1001", "ABC-1", "2026-03-11T10:00:00.000+0000", "First")
        issue2 = FakeIssue("1002", "ABC-2", "2026-03-11T10:01:00.000+0000", "Second")
        fake_client = FakeClient([[issue1, issue2], []])

        source = JiraSource(config=cfg, client=fake_client)
        events = source.fetch_new_events(last_seen=None)

        self.assertEqual([e.ticket_key for e in events], ["ABC-1", "ABC-2"])
        self.assertEqual(events[0].summary, "First")
        self.assertEqual(events[0].project_key, "DEMO")
        self.assertIn('project = "ABC"', str(fake_client.calls[0]["jql"]))

    def test_jira_source_filters_events_strictly_newer_than_last_seen(self) -> None:
        cfg = JiraConfig(
            server="https://example.atlassian.net",
            email="a@b.com",
            api_token="token",
            project_key="ABC",
            page_size=5,
        )

        fake_client = FakeClient(
            [[
                FakeIssue("1001", "ABC-1", "2026-03-11T10:00:00.000+0000", "Old"),
                FakeIssue("1002", "ABC-2", "2026-03-11T10:05:00.000+0000", "New"),
            ]]
        )
        source = JiraSource(config=cfg, client=fake_client)

        events = source.fetch_new_events(last_seen=JiraSource._parse_created("2026-03-11T10:00:00.000+0000"))

        self.assertEqual([e.ticket_key for e in events], ["ABC-2"])


if __name__ == "__main__":
    unittest.main()

