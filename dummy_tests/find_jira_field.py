from __future__ import annotations

import argparse
import os

from jira import JIRA

from validation_enrichment.env_loader import load_env_file


def _get_env(name: str) -> str:
    value = (os.getenv(name) or "").strip()
    if not value:
        raise ValueError(f"Missing required env var: {name}")
    return value


def main() -> None:
    load_env_file(".env")

    parser = argparse.ArgumentParser(description="Find Jira custom field IDs by name")
    parser.add_argument(
        "--contains",
        default="Version",
        help="Case-insensitive text to match in Jira field names",
    )
    parser.add_argument(
        "--custom-only",
        action="store_true",
        help="Show only customfield_* entries",
    )
    args = parser.parse_args()

    base_url = (os.getenv("JIRA_BASE_URL") or os.getenv("JIRA_SERVER") or "").strip()
    if not base_url:
        raise ValueError("Missing required env var: JIRA_BASE_URL")

    email = _get_env("JIRA_EMAIL")
    token = _get_env("JIRA_API_TOKEN")

    jira = JIRA(server=base_url, basic_auth=(email, token))
    fields = jira.fields()

    needle = args.contains.strip().lower()
    matches = []
    for field in fields:
        field_id = str(field.get("id", ""))
        field_name = str(field.get("name", ""))
        if args.custom_only and not field_id.startswith("customfield_"):
            continue
        if needle in field_name.lower():
            matches.append((field_id, field_name))

    if not matches:
        print(f"No fields matched: '{args.contains}'")
        return

    print(f"Matched fields for '{args.contains}':")
    for field_id, field_name in matches:
        print(f"- {field_id}: {field_name}")


if __name__ == "__main__":
    main()

