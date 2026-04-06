from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from agent2.category_prompt_config import CategoryPromptConfig
from agent2.env_loader import load_env_file
from agent2.jira_source import JiraTicketSource
from agent2.openai_categorizer import OpenAICategorizer
from agent2.processor import ValidationEnrichmentProcessor
from agent2.rules import RuleSet


def _env_or_default(key: str, default: str) -> str:
    value = os.getenv(key, "").strip()
    return value or default


def _build_source_from_env():
    source_type = os.getenv("AGENT2_SOURCE", "jira").strip().lower()
    if source_type == "jira":
        return JiraTicketSource.from_env()
    raise ValueError(f"Unsupported AGENT2_SOURCE '{source_type}'. Supported: jira")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agent2: Validation and Enrichment")
    parser.add_argument("--issue-key", default=os.getenv("AGENT2_INPUT_ISSUE_KEY", ""), help="Issue key from Agent1")
    return parser.parse_args()


def main() -> None:
    load_env_file(".env")
    args = _parse_args()
    issue_key = (args.issue_key or "").strip()
    if not issue_key:
        raise ValueError("Issue key is required. Provide --issue-key or AGENT2_INPUT_ISSUE_KEY in .env")

    source = _build_source_from_env()
    rules = RuleSet.from_file(_env_or_default("AGENT2_RULES_FILE", "agent2/rules.json"))
    prompt_config = CategoryPromptConfig.from_file(
        _env_or_default("AGENT2_PROMPTS_FILE", "agent2/category_prompts.txt")
    )
    categorizer = OpenAICategorizer(rules)
    processor = ValidationEnrichmentProcessor(
        source=source,
        rules=rules,
        categorizer=categorizer,
        prompt_config=prompt_config,
    )

    attachment_root = _env_or_default("AGENT2_ATTACHMENT_ROOT", f"agent2/attachments/{issue_key}")
    result = processor.process_issue(issue_key=issue_key, attachment_root=attachment_root)

    output_path = Path(_env_or_default("AGENT2_OUTPUT_FILE", f"agent2/output/{issue_key}.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    attachment_paths = [str(Path(path).resolve()) for path in result.attachment_paths]
    stats = getattr(source, "last_attachment_stats", {}) if source is not None else {}
    downloaded_count = int(stats.get("downloaded_count", 0) or 0) if isinstance(stats, dict) else 0
    reused_count = int(stats.get("reused_count", 0) or 0) if isinstance(stats, dict) else 0
    if attachment_paths and downloaded_count == 0 and reused_count > 0:
        attachment_status_message = "Existing attachments loaded (no new downloads)."
    elif downloaded_count > 0 and reused_count > 0:
        attachment_status_message = (
            f"Downloaded {downloaded_count} new attachment(s), reused {reused_count} existing attachment(s)."
        )
    elif downloaded_count > 0:
        attachment_status_message = f"Downloaded {downloaded_count} attachment(s)."
    else:
        attachment_status_message = "No attachments available on this ticket."

    raw_ticket = result.raw_ticket if isinstance(result.raw_ticket, dict) else {}
    output_payload = {
        "Summary": str(raw_ticket.get("summary", "")).strip(),
        "Issue key": str(raw_ticket.get("issue_key", "") or result.issue_key).strip(),
        "Description": str(raw_ticket.get("description", "")).strip(),
        "Organizations": raw_ticket.get("organizations", []) if isinstance(raw_ticket, dict) else [],
        "Comments": raw_ticket.get("comments", []) if isinstance(raw_ticket, dict) else [],
        "Attachment Paths": attachment_paths,
        "Attachment Status Message": attachment_status_message,
        "Missing Information": result.missing_information,
    }
    output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")

    print(f"[agent2] Issue processed: {issue_key}")
    print(f"[agent2] Output JSON: {output_path.resolve()}")
    print(f"[agent2] Attachments downloaded: {len(result.attachment_paths)}")
    print(f"[agent2] {attachment_status_message}")
    for attachment_path in output_payload["Attachment Paths"]:
        print(f"[agent2] Attachment path: {attachment_path}")


if __name__ == "__main__":
    main()
