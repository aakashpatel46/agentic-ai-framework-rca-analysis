from __future__ import annotations

import argparse
import os
import re
import smtplib
import ssl
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path

from validation_enrichment.env_loader import load_env_file
from streamlit_app import (
    _map_agent2_to_agent3_ticket,
    run_agent2,
    run_agent3,
    run_agent4,
    run_agent5,
)


def _safe_issue_key(value: str) -> str:
    return "".join([c if c.isalnum() or c in "-_." else "_" for c in value.strip()]) or "ticket"


def _extract_first_timestamp(text: str) -> str:
    value = (text or "").strip()
    if not value:
        return ""
    patterns = [
        r"\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?\b",
        r"\b\d{2}/[A-Za-z]{3}/\d{2}\s+\d{1,2}:\d{2}\s*[AP]M\b",
    ]
    for pattern in patterns:
        m = re.search(pattern, value)
        if m:
            return m.group(0)
    return ""


def _keywords(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z0-9_]+", (text or "").lower())
    stop = {"the", "and", "for", "with", "that", "this", "from", "have", "has", "was", "are", "not"}
    return {w for w in words if len(w) > 3 and w not in stop}


def _similarity_reason(current_summary: str, current_desc: str, row: dict) -> str:
    sim_summary = str(row.get("summary", "")).strip()
    sim_desc = str(row.get("description", "")).strip()
    sim_score = row.get("score", None)
    overlap = sorted(list((_keywords(current_summary + " " + current_desc)) & (_keywords(sim_summary + " " + sim_desc))))[:4]
    parts: list[str] = []
    if sim_score is not None:
        parts.append(f"vector score {sim_score}")
    if overlap:
        parts.append(f"shared context: {', '.join(overlap)}")
    ai = str(row.get("ai_summary", "")).strip()
    if ai:
        parts.append(ai[:140] + ("..." if len(ai) > 140 else ""))
    return "; ".join(parts) if parts else "Similar historical pattern."


def _non_technical_summary(issue_summary: str, rca_summary: str, root_cause: str) -> str:
    s1 = (issue_summary or "").strip()
    s2 = (rca_summary or "").strip()
    s3 = (root_cause or "").strip()
    lines = []
    if s1:
        lines.append(f"The issue is about: {s1[:220]}")
    if s2:
        lines.append(f"What happened: {s2[:280]}")
    if s3:
        lines.append(f"Why it happened: {s3[:280]}")
    return "\n".join(lines[:3]) or "Summary unavailable."


def _build_rca_message(
    issue_key: str,
    agent2_output: dict,
    agent3_output: dict,
    agent4_output: dict,
    agent5_output: dict,
) -> str:
    ticket = agent4_output.get("input_ticket", {}) if isinstance(agent4_output, dict) else {}
    rca = agent4_output.get("detailed_rca_and_fix", {}) if isinstance(agent4_output, dict) else {}
    risk = agent5_output.get("detailed_risk_analysis", {}) if isinstance(agent5_output, dict) else {}
    agent3_analysis = agent3_output.get("detailed_analysis", {}) if isinstance(agent3_output, dict) else {}

    issue_summary = str(ticket.get("Summary", ticket.get("summary", "N/A"))).strip()
    issue_description = str(ticket.get("Description", ticket.get("description", "N/A"))).strip()
    executive_summary = str(rca.get("executive_summary", "N/A")).strip()
    root_cause = str(rca.get("root_cause_analysis", "N/A")).strip()
    recommended_fix = str(rca.get("recommended_fix_path", "N/A")).strip()
    rca_confidence = str(rca.get("confidence", "N/A")).strip()
    risk_level = str(risk.get("overall_risk_level", "N/A")).strip()
    risk_confidence = str(risk.get("confidence", "N/A")).strip()
    go_no_go = str(risk.get("go_no_go_recommendation", "N/A")).strip()

    similar_info_summary = str(agent3_analysis.get("executive_summary", "N/A")).strip()

    missing_info = (
        agent2_output.get("Missing Information", [])
        if isinstance(agent2_output, dict)
        else []
    )
    if not missing_info and isinstance(agent2_output, dict):
        missing_info = agent2_output.get("missing_information", [])

    root_cause_factors = rca.get("root_cause_factors", [])
    evidence_used = rca.get("evidence_used", [])
    possible_fixes = rca.get("possible_fixes", [])
    risk_drivers = risk.get("risk_drivers", [])
    mitigation_prechecks = risk.get("mitigation_prechecks", [])
    assumptions = risk.get("assumptions_and_unknowns", [])
    risk_dimensions = risk.get("risk_dimensions", {})

    similar = agent3_output.get("similar_tickets_top3", []) or agent3_output.get("similar_tickets", [])
    logs = agent3_output.get("logs", []) if isinstance(agent3_output, dict) else []
    yml_files = agent3_output.get("yml_files", []) if isinstance(agent3_output, dict) else []
    other_attachments = agent3_output.get("other_attachments", []) if isinstance(agent3_output, dict) else []

    summary_block = _non_technical_summary(issue_summary, executive_summary, root_cause)

    lines = [
        f"RCA Detailed Report for {issue_key}",
        "",
        f"Issue Key: {issue_key}",
        "",
        "Summary:",
        summary_block,
        "",
        "Missing Information:",
    ]
    if isinstance(missing_info, list) and missing_info:
        for item in missing_info:
            lines.append(f"- {item}")
    else:
        lines.append("- None")

    lines.extend(["", "Similar Tickets:"])
    if isinstance(similar, list) and similar:
        lines.append(f"- Information from similar tickets: {similar_info_summary}")
        current_desc = issue_description or issue_summary
        for idx, row in enumerate(similar, start=1):
            if not isinstance(row, dict):
                continue
            sim_key = str(row.get("issue_key", f"TICKET-{idx}")).strip()
            lines.append(f"- {sim_key}: {_similarity_reason(issue_summary, current_desc, row)}")
            lines.append("  Raw Information:")
            lines.append(f"    Summary: {row.get('summary', 'N/A')}")
            lines.append(f"    Description: {row.get('description', 'N/A')}")
            lines.append(f"    AI Summary: {row.get('ai_summary', 'N/A')}")
            lines.append(f"    Score: {row.get('score', 'N/A')}")
            lines.append(f"    Updated: {row.get('updated', 'N/A')}")
            comments = row.get("comments", [])
            if isinstance(comments, list) and comments:
                lines.append("    Comments:")
                for c in comments[:5]:
                    lines.append(f"      - {str(c).strip()}")
            elif isinstance(comments, str) and comments.strip():
                lines.append("    Comments:")
                lines.append(f"      - {comments.strip()}")
            else:
                lines.append("    Comments: Not available in indexed metadata.")
    else:
        lines.append("- No similar-ticket context available.")

    lines.extend(["", "Attachment Analysis:"])
    any_attachment = False
    for items, group in ((logs, "log"), (yml_files, "yml"), (other_attachments, "other")):
        if not isinstance(items, list):
            continue
        for idx, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                continue
            any_attachment = True
            p = str(item.get("path", "")).strip()
            name = Path(p).name if p else f"{group}_{idx}"
            excerpt = str(item.get("excerpt", "")).strip()
            note = str(item.get("note", "")).strip()
            ts = _extract_first_timestamp(excerpt)
            if group == "log":
                base = f"- {name}: log evidence"
                if ts:
                    base += f" at {ts}"
                lines.append(base)
                if excerpt:
                    lines.append(f"  Evidence: {excerpt[:300]}{'...' if len(excerpt) > 300 else ''}")
            elif group == "yml":
                lines.append(f"- {name}: configuration evidence found.")
                if excerpt:
                    lines.append(f"  Evidence: {excerpt[:300]}{'...' if len(excerpt) > 300 else ''}")
            else:
                image_analysis = item.get("image_analysis", {}) if isinstance(item, dict) else {}
                if isinstance(image_analysis, dict) and str(image_analysis.get("summary", "")).strip():
                    summary = str(image_analysis.get("summary", "")).strip()
                    extracted = str(image_analysis.get("extracted_text", "")).strip()
                    timestamps = image_analysis.get("timestamps", [])
                    ts_value = str(timestamps[0]).strip() if isinstance(timestamps, list) and timestamps else ""
                    if ts_value:
                        lines.append(f"- {name}: {summary} (timestamp: {ts_value})")
                    else:
                        lines.append(f"- {name}: {summary}")
                    if extracted:
                        lines.append(f"  Evidence: {extracted[:300]}{'...' if len(extracted) > 300 else ''}")
                else:
                    lines.append(f"- {name}: {note or 'Attachment inspected for metadata/context.'}")
    if not any_attachment:
        lines.append("- No analyzable attachment evidence found.")

    lines.extend(
        [
            "",
            "RCA Summary:",
            f"- Executive Summary: {executive_summary}",
            f"- Root Cause: {root_cause}",
            "",
            "Root Cause Factors:",
        ]
    )

    if isinstance(root_cause_factors, list) and root_cause_factors:
        for item in root_cause_factors:
            lines.append(f"- {item}")
    else:
        lines.append("- N/A")

    lines.extend(["", "Evidence Used:"])
    if isinstance(evidence_used, list) and evidence_used:
        for item in evidence_used:
            lines.append(f"- {item}")
    else:
        lines.append("- N/A")

    lines.extend(["", "Recommended Fix:", f"- {recommended_fix}", "", "Possible Fix Options:"])
    if isinstance(possible_fixes, list) and possible_fixes:
        for idx, fix in enumerate(possible_fixes, start=1):
            if not isinstance(fix, dict):
                continue
            title = str(fix.get("title", f"Fix {idx}")).strip()
            rationale = str(fix.get("rationale", "N/A")).strip()
            fix_risk = str(fix.get("risk_level", "N/A")).strip()
            lines.append(f"- {title} (Risk: {fix_risk})")
            lines.append(f"  Rationale: {rationale}")
    else:
        lines.append("- N/A")

    lines.extend(
        [
            "",
            "Risk Analysis:",
            f"- Overall Risk Level: {risk_level}",
            f"- Go/No-Go Recommendation: {go_no_go}",
            f"- Risk Confidence: {risk_confidence}",
            f"- RCA Confidence: {rca_confidence}",
        ]
    )

    if isinstance(risk_dimensions, dict) and risk_dimensions:
        lines.append("- Risk Dimensions:")
        for key, value in risk_dimensions.items():
            lines.append(f"  - {key}: {value}")

    lines.extend(["", "RISK DRIVERS"])
    if isinstance(risk_drivers, list) and risk_drivers:
        for item in risk_drivers:
            lines.append(f"- {item}")
    else:
        lines.append("- N/A")

    lines.extend(["", "MITIGATION PRECHECKS"])
    if isinstance(mitigation_prechecks, list) and mitigation_prechecks:
        for item in mitigation_prechecks:
            lines.append(f"- {item}")
    else:
        lines.append("- N/A")

    lines.extend(["", "ASSUMPTIONS / UNKNOWNS"])
    if isinstance(assumptions, list) and assumptions:
        for item in assumptions:
            lines.append(f"- {item}")
    else:
        lines.append("- N/A")

    return "\n".join(lines)


def _get_required_env(name: str) -> str:
    value = (os.getenv(name, "") or "").strip().strip('"').strip("'")
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _send_email(issue_key: str, message: str) -> str:
    smtp_host = _get_required_env("SMTP_HOST")
    smtp_port = int(_get_required_env("SMTP_PORT"))
    smtp_username = _get_required_env("SMTP_USERNAME")
    smtp_password = _get_required_env("SMTP_PASSWORD")
    email_from = _get_required_env("ALERT_EMAIL_FROM")
    email_to = _get_required_env("ALERT_EMAIL_TO")
    use_tls = (os.getenv("SMTP_USE_TLS", "true") or "").strip().lower() in {"1", "true", "yes", "y"}

    msg = EmailMessage()
    msg["Subject"] = f"RCA Summary - {issue_key}"
    msg["From"] = email_from
    msg["To"] = email_to
    msg.set_content(message)

    if use_tls:
        context = ssl.create_default_context()
        with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
            server.ehlo()
            server.starttls(context=context)
            server.ehlo()
            server.login(smtp_username, smtp_password)
            server.send_message(msg)
    else:
        with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=30) as server:
            server.login(smtp_username, smtp_password)
            server.send_message(msg)
    return email_to


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual RCA runner for a Jira issue key")
    parser.add_argument("--issue-key", required=True, help="Jira issue key (example: JMCH-1802)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    issue_key = args.issue_key.strip()
    if not issue_key:
        raise ValueError("Issue key cannot be empty.")

    load_env_file(".env")

    print(f"[runner] Starting pipeline for {issue_key}")

    agent2_output, agent2_path, _ = run_agent2(issue_key)
    print(f"[runner] Validation & enrichment done: {agent2_path.resolve()}")

    ticket_payload = _map_agent2_to_agent3_ticket(agent2_output)
    agent3_output, agent3_path = run_agent3(ticket_payload)
    print(f"[runner] Multi-source analysis done: {agent3_path.resolve()}")

    agent4_output, agent4_path = run_agent4(ticket_payload, agent3_output)
    print(f"[runner] RCA synthesis done: {agent4_path.resolve()}")

    agent5_output, agent5_path = run_agent5(agent4_output)
    print(f"[runner] Risk reporting done: {agent5_path.resolve()}")

    message = _build_rca_message(issue_key, agent2_output, agent3_output, agent4_output, agent5_output)

    output_dir = Path("framework/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{_safe_issue_key(issue_key)}_rca_message.txt"
    out_file.write_text(message, encoding="utf-8")
    print(f"[runner] RCA message prepared: {out_file.resolve()}")

    sent_to = _send_email(issue_key, message)
    print(f"[runner] Email sent successfully to {sent_to}.")
    print(f"[runner] Completed at {datetime.now(timezone.utc).replace(microsecond=0).isoformat()}")


if __name__ == "__main__":
    main()
