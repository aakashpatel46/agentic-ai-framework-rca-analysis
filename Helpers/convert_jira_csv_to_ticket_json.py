# python Helpers\convert_jira_csv_to_ticket_json.py --input Helpers/files/Jira.csv --output Helpers/files/JMCH_tickets_raw_json.json


from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

DEFAULT_INPUT = "Helpers/files/Jira.csv"
DEFAULT_OUTPUT = "Helpers/files/JMCH_tickets_raw_json.json"


def clean(value: str) -> str:
    return (value or "").strip()


def values_for_header(headers: list[str], row: list[str], target: str) -> list[str]:
    out: list[str] = []
    for i, header in enumerate(headers):
        if header == target and i < len(row):
            v = clean(row[i])
            if v:
                out.append(v)
    return out


def first_for_header(headers: list[str], row: list[str], target: str) -> str:
    vals = values_for_header(headers, row, target)
    return vals[0] if vals else ""


def to_ticket(headers: list[str], row: list[str]) -> dict[str, object]:
    summary = first_for_header(headers, row, "Summary")
    issue_key = first_for_header(headers, row, "Issue key")
    description = first_for_header(headers, row, "Description")

    organizations = values_for_header(headers, row, "Custom field (Organizations)")
    comments = values_for_header(headers, row, "Comment")

    return {
        "Summary": summary,
        "Issue key": issue_key,
        "Description": description,
        "Organizations": organizations,
        "Comments": comments,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Jira CSV export into ticket JSON expected by ticket upsert scripts."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help=f"Input CSV path (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help=f"Output JSON path (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    tickets: list[dict[str, object]] = []
    seen_keys: set[str] = set()

    with input_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        headers = next(reader, None)
        if not headers:
            raise ValueError("CSV appears empty (no header row).")

        for row in reader:
            ticket = to_ticket(headers, row)
            issue_key = str(ticket.get("Issue key", "")).strip()
            if not issue_key or issue_key in seen_keys:
                continue
            seen_keys.add(issue_key)
            tickets.append(ticket)

    tickets.sort(key=lambda t: str(t.get("Issue key", "")))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(tickets, f, ensure_ascii=False, indent=2)

    print(f"Converted tickets: {len(tickets)}")
    print(f"Output written: {output_path.resolve()}")


if __name__ == "__main__":
    main()
