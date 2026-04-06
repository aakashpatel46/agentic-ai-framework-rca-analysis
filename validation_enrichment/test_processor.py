from __future__ import annotations

import unittest
from datetime import datetime, timezone

from validation_enrichment.category_prompt_config import CategoryPromptConfig
from validation_enrichment.models import AttachmentMeta, RawTicket
from validation_enrichment.processor import ValidationEnrichmentProcessor
from validation_enrichment.rules import RuleSet
from validation_enrichment.source import TicketSource


class FakeSource(TicketSource):
    def __init__(self, description: str = "Error observed in payment flow", steps: str = "", version: str = "") -> None:
        self._description = description
        self._steps = steps
        self._version = version

    @property
    def name(self) -> str:
        return "fake"

    def fetch_issue(self, issue_key: str) -> RawTicket:
        return RawTicket(
            source="fake",
            issue_id="1",
            issue_key=issue_key,
            summary="Bug in checkout",
            description=self._description,
            steps_to_reproduce=self._steps,
            affects_version=self._version,
            issue_type="Bug",
            status="Open",
            priority="High",
            reporter="A User",
            assignee="",
            labels=["payments"],
            created_at=datetime(2026, 3, 11, 10, 0, tzinfo=timezone.utc),
            updated_at=datetime(2026, 3, 11, 10, 5, tzinfo=timezone.utc),
            attachments=[
                AttachmentMeta(
                    id="att1",
                    filename="error_logs.txt",
                    mime_type="text/plain",
                    size=22,
                    content_url="https://example/att1",
                )
            ],
            raw={"id": "1"},
        )

    def download_attachments(self, ticket: RawTicket, destination_dir: str) -> list[str]:
        return [f"{destination_dir}/error_logs.txt"]


class FakeCategorizer:
    def __init__(self, category: str) -> None:
        self._category = category

    def categorize(self, ticket: RawTicket) -> str:
        return self._category

    def enrich_missing_information(
        self,
        ticket: RawTicket,
        category: str,
        missing_information: list[str],
        secondary_prompt: str,
    ) -> dict:
        return {
            "resolved_information": {},
            "follow_up_questions": [],
            "recommended_actions": [],
            "notes": "fake",
        }


def _prompt_config() -> CategoryPromptConfig:
    return CategoryPromptConfig(
        default_secondary_prompt="default",
        category_secondary_prompts={"bug": "bug prompt"},
    )


class ProcessorTests(unittest.TestCase):
    def test_process_issue_detects_missing_attachment_requirement(self) -> None:
        rules = RuleSet(
            {
                "bug": ["summary", "description", "priority", "attachment:screenshot"],
                "other": ["summary"],
            }
        )
        processor = ValidationEnrichmentProcessor(
            source=FakeSource(),
            rules=rules,
            categorizer=FakeCategorizer("bug"),
            prompt_config=_prompt_config(),
        )

        result = processor.process_issue(issue_key="ABC-1", attachment_root="validation_enrichment/attachments/ABC-1")

        self.assertEqual(result.category, "bug")
        self.assertIn("summary", result.available_information)
        self.assertIn("attachment", result.available_information)
        self.assertEqual(result.missing_information, ["attachment:screenshot"])
        self.assertTrue(result.secondary_openai_called)

    def test_process_issue_detects_steps_to_reproduce_and_logs(self) -> None:
        description = (
            "Steps to reproduce:\n"
            "1. Open checkout page\n"
            "2. Add item\n"
            "3. Submit order\n"
            "Actual result: Error appears\n"
        )
        rules = RuleSet(
            {
                "bug": ["summary", "description", "Logs", "Reproduction steps"],
                "other": ["summary"],
            }
        )
        processor = ValidationEnrichmentProcessor(
            source=FakeSource(description=description),
            rules=rules,
            categorizer=FakeCategorizer("bug"),
            prompt_config=_prompt_config(),
        )

        result = processor.process_issue(issue_key="ABC-2", attachment_root="validation_enrichment/attachments/ABC-2")

        self.assertIn("reproduction steps", result.available_information)
        self.assertEqual(result.missing_information, [])
        self.assertFalse(result.secondary_openai_called)

    def test_logs_rule_passes_for_dot_log_attachment(self) -> None:
        rules = RuleSet(
            {
                "bug": ["summary", "description", "Logs"],
                "other": ["summary"],
            }
        )
        processor = ValidationEnrichmentProcessor(
            source=FakeSource(description="Some description without steps"),
            rules=rules,
            categorizer=FakeCategorizer("bug"),
            prompt_config=_prompt_config(),
        )

        result = processor.process_issue(issue_key="ABC-3", attachment_root="validation_enrichment/attachments/ABC-3")

        self.assertIn("logs", result.available_information)
        self.assertEqual(result.missing_information, [])

    def test_repro_steps_from_custom_field(self) -> None:
        rules = RuleSet(
            {
                "bug": ["summary", "Reproduction steps"],
                "other": ["summary"],
            }
        )
        processor = ValidationEnrichmentProcessor(
            source=FakeSource(description="No explicit steps in description", steps="1. Open app\n2. Reproduce bug"),
            rules=rules,
            categorizer=FakeCategorizer("bug"),
            prompt_config=_prompt_config(),
        )

        result = processor.process_issue(issue_key="ABC-4", attachment_root="validation_enrichment/attachments/ABC-4")

        self.assertIn("reproduction steps", result.available_information)
        self.assertEqual(result.missing_information, [])

    def test_affects_version_from_custom_field(self) -> None:
        rules = RuleSet(
            {
                "bug": ["summary", "Version"],
                "other": ["summary"],
            }
        )
        processor = ValidationEnrichmentProcessor(
            source=FakeSource(version="250.0.2"),
            rules=rules,
            categorizer=FakeCategorizer("bug"),
            prompt_config=_prompt_config(),
        )

        result = processor.process_issue(issue_key="ABC-5", attachment_root="validation_enrichment/attachments/ABC-5")

        self.assertIn("version", result.available_information)
        self.assertEqual(result.missing_information, [])


if __name__ == "__main__":
    unittest.main()

