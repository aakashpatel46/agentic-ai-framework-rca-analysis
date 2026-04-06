from __future__ import annotations

import json
import os
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st
from openai import OpenAI

from agent1.jira_source import JiraSource
from agent2.category_prompt_config import CategoryPromptConfig
from agent2.env_loader import load_env_file
from agent2.jira_source import JiraTicketSource
from agent2.openai_categorizer import OpenAICategorizer
from agent2.processor import ValidationEnrichmentProcessor
from agent2.rules import RuleSet
from agent3.agent3 import Agent3Analyzer
from agent4.agent4 import Agent4RcaSynthesizer
from agent5.agent5 import Agent5RiskAnalyzer


def _env_or_default(key: str, default: str) -> str:
    value = os.getenv(key, "").strip()
    return value or default


def _safe_issue_key(value: str) -> str:
    return "".join([c if c.isalnum() or c in "-_." else "_" for c in value.strip()]) or "ticket"


def _json_dump(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False, default=str)


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json_dump(payload), encoding="utf-8")
    return path


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def _append_log(message: str) -> None:
    st.session_state["framework_logs"].append(message)


def _queue_issue_key_update(value: str) -> None:
    st.session_state["pending_issue_key_input"] = value


def _normalize_usage_dict(usage: dict[str, Any] | None) -> dict[str, int]:
    if not isinstance(usage, dict):
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    input_tokens = int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
    output_tokens = int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0)
    total_tokens = int(usage.get("total_tokens", input_tokens + output_tokens) or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _record_token_event(agent: str, call_name: str, usage: dict[str, Any] | None) -> None:
    item = _normalize_usage_dict(usage)
    st.session_state["token_events"].append(
        {
            "agent": agent,
            "call": call_name,
            "input_tokens": item["input_tokens"],
            "output_tokens": item["output_tokens"],
            "total_tokens": item["total_tokens"],
        }
    )


def _record_agent_usage(agent: str, payload: dict[str, Any] | None) -> None:
    if not isinstance(payload, dict):
        return
    if agent == "agent2":
        usage = payload.get("token_usage", {})
        if isinstance(usage, dict):
            _record_token_event(agent, "categorization", usage.get("categorization", {}))
            if bool(payload.get("secondary_openai_called")) or bool(payload.get("Secondary OpenAI Called")):
                _record_token_event(agent, "secondary_enrichment", usage.get("secondary_enrichment", {}))
        return
    if agent == "agent3":
        token_usage = payload.get("token_usage", {})
        calls = token_usage.get("calls", []) if isinstance(token_usage, dict) else []
        if isinstance(calls, list):
            for row in calls:
                if isinstance(row, dict):
                    _record_token_event(agent, str(row.get("call", "unknown")), row)
        return
    if agent == "agent4":
        detailed = payload.get("detailed_rca_and_fix", {})
        if isinstance(detailed, dict):
            _record_token_event(agent, "rca_synthesis", detailed.get("token_usage", {}))
        return
    if agent == "agent5":
        detailed = payload.get("detailed_risk_analysis", {})
        if isinstance(detailed, dict):
            _record_token_event(agent, "risk_synthesis", detailed.get("token_usage", {}))
        return


def _token_totals() -> dict[str, int]:
    in_total = sum(int(x.get("input_tokens", 0) or 0) for x in st.session_state["token_events"])
    out_total = sum(int(x.get("output_tokens", 0) or 0) for x in st.session_state["token_events"])
    total = sum(int(x.get("total_tokens", 0) or 0) for x in st.session_state["token_events"])
    return {"input_tokens": in_total, "output_tokens": out_total, "total_tokens": total}


def _discover_previous_issue_keys() -> list[str]:
    candidates = [
        Path("framework/output"),
        Path("agent5/output"),
        Path("agent4/output"),
        Path("agent3/output"),
        Path("agent2/output"),
        Path("agent1/output"),
    ]
    keys: set[str] = set()
    for base in candidates:
        if not base.exists():
            continue
        for item in base.glob("*.json"):
            keys.add(item.stem.strip())
    return sorted([k for k in keys if k], reverse=True)


def _next_chat_file_path(ticket_key: str) -> Path:
    chats_dir = Path("chats")
    chats_dir.mkdir(parents=True, exist_ok=True)
    safe_key = _safe_issue_key(ticket_key)
    base = chats_dir / f"{safe_key}.json"
    if not base.exists():
        return base
    i = 2
    while True:
        candidate = chats_dir / f"{safe_key}({i}).json"
        if not candidate.exists():
            return candidate
        i += 1


def _discover_chat_files(ticket_key: str | None = None) -> list[Path]:
    chats_dir = Path("chats")
    if not chats_dir.exists():
        return []
    files = sorted(chats_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not ticket_key:
        return files
    safe_key = _safe_issue_key(ticket_key)
    out: list[Path] = []
    for p in files:
        stem = p.stem
        if stem == safe_key or stem.startswith(f"{safe_key}("):
            out.append(p)
    return out


def _save_chat_session(ticket_key: str, final_output: dict[str, Any], qa_history: list[dict[str, Any]]) -> Path | None:
    if not qa_history:
        return None
    chat_path_text = st.session_state.get("active_chat_file_path", "")
    safe_key = _safe_issue_key(ticket_key)
    if chat_path_text:
        chat_path = Path(chat_path_text)
        stem = chat_path.stem
        if not (stem == safe_key or stem.startswith(f"{safe_key}(")):
            chat_path = _next_chat_file_path(ticket_key)
            st.session_state["active_chat_file_path"] = str(chat_path.resolve())
    else:
        chat_path = _next_chat_file_path(ticket_key)
        st.session_state["active_chat_file_path"] = str(chat_path.resolve())

    payload = {
        "ticket_key": ticket_key,
        "saved_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "final_output": final_output,
        "qa_history": qa_history,
    }
    _write_json(chat_path, payload)
    return chat_path


def _load_chat_session(chat_path: Path) -> dict[str, Any]:
    value = _read_json_if_exists(chat_path)
    if value is None:
        raise RuntimeError(f"Chat file is invalid or missing: {chat_path}")
    return value


def _extract_jira_comments(raw_ticket: dict[str, Any]) -> list[str]:
    raw = raw_ticket.get("raw", {}) if isinstance(raw_ticket, dict) else {}
    fields = raw.get("fields", {}) if isinstance(raw, dict) else {}
    comment_obj = fields.get("comment", {}) if isinstance(fields, dict) else {}
    comments = comment_obj.get("comments", []) if isinstance(comment_obj, dict) else []
    out: list[str] = []
    for item in comments:
        if not isinstance(item, dict):
            continue
        body = item.get("body", "")
        if isinstance(body, str) and body.strip():
            out.append(body.strip())
    return out


def _map_agent2_to_agent3_ticket(agent2_output: dict[str, Any]) -> dict[str, Any]:
    # Preferred Agent2 handoff payload.
    issue_key = str(agent2_output.get("Issue key", "")).strip()
    summary = str(agent2_output.get("Summary", "")).strip()
    description = str(agent2_output.get("Description", "")).strip()
    organizations = agent2_output.get("Organizations", [])
    comments = agent2_output.get("Comments", [])
    attachment_paths = agent2_output.get("Attachment Paths", [])
    if issue_key or summary or description:
        return {
            "Issue key": issue_key,
            "Summary": summary,
            "Description": description,
            "Organizations": organizations if isinstance(organizations, list) else [],
            "Comments": comments if isinstance(comments, list) else [],
            "Attachment Paths": attachment_paths if isinstance(attachment_paths, list) else [],
        }

    # Backward compatibility for older payloads that contained raw_ticket.
    raw_ticket = agent2_output.get("raw_ticket", {}) if isinstance(agent2_output, dict) else {}
    if not isinstance(raw_ticket, dict):
        raw_ticket = {}

    attachments = raw_ticket.get("attachments", [])
    attachment_names = []
    if isinstance(attachments, list):
        for item in attachments:
            if isinstance(item, dict):
                name = str(item.get("filename", "")).strip()
                if name:
                    attachment_names.append(name)

    comments = _extract_jira_comments(raw_ticket)
    mapped = {
        "Issue key": str(raw_ticket.get("issue_key", "")).strip(),
        "Summary": str(raw_ticket.get("summary", "")).strip(),
        "Description": str(raw_ticket.get("description", "")).strip(),
        "Priority": str(raw_ticket.get("priority", "")).strip(),
        "Issue Type": str(raw_ticket.get("issue_type", "")).strip(),
        "Status": str(raw_ticket.get("status", "")).strip(),
        "Reporter": str(raw_ticket.get("reporter", "")).strip(),
        "Assignee": str(raw_ticket.get("assignee", "")).strip(),
        "Labels": raw_ticket.get("labels", []) if isinstance(raw_ticket.get("labels", []), list) else [],
        "Steps to Reproduce": str(raw_ticket.get("steps_to_reproduce", "")).strip(),
        "Affects Version/s": str(raw_ticket.get("affects_version", "")).strip(),
        "Comments": comments,
        "Attachment Names": attachment_names,
    }
    if not mapped["Issue key"]:
        mapped["Issue key"] = str(agent2_output.get("issue_key", "")).strip()
    return mapped


def run_agent2(issue_key: str) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    source = JiraTicketSource.from_env()
    rules = RuleSet.from_file(_env_or_default("AGENT2_RULES_FILE", "agent2/rules.json"))
    prompt_config = CategoryPromptConfig.from_file(_env_or_default("AGENT2_PROMPTS_FILE", "agent2/category_prompts.txt"))
    categorizer = OpenAICategorizer(rules)
    processor = ValidationEnrichmentProcessor(
        source=source,
        rules=rules,
        categorizer=categorizer,
        prompt_config=prompt_config,
    )

    attachment_root = _env_or_default("AGENT2_ATTACHMENT_ROOT", f"agent2/attachments/{issue_key}")
    result = processor.process_issue(issue_key=issue_key, attachment_root=attachment_root)
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
    payload = {
        "Summary": str(raw_ticket.get("summary", "")).strip(),
        "Issue key": str(raw_ticket.get("issue_key", "") or result.issue_key).strip(),
        "Description": str(raw_ticket.get("description", "")).strip(),
        "Organizations": raw_ticket.get("organizations", []) if isinstance(raw_ticket, dict) else [],
        "Comments": raw_ticket.get("comments", []) if isinstance(raw_ticket, dict) else [],
        "Attachment Paths": attachment_paths,
        "Attachment Status Message": attachment_status_message,
        "Missing Information": result.missing_information,
    }

    token_usage = {
        "categorization": categorizer.last_categorize_usage,
        "secondary_enrichment": categorizer.last_secondary_usage,
    }
    usage_totals = _normalize_usage_dict(categorizer.last_categorize_usage)
    secondary_totals = _normalize_usage_dict(categorizer.last_secondary_usage)
    token_usage["session_totals"] = {
        "input_tokens": usage_totals["input_tokens"] + secondary_totals["input_tokens"],
        "output_tokens": usage_totals["output_tokens"] + secondary_totals["output_tokens"],
        "total_tokens": usage_totals["total_tokens"] + secondary_totals["total_tokens"],
    }
    payload["token_usage"] = token_usage

    output_path = Path(_env_or_default("AGENT2_OUTPUT_FILE", f"agent2/output/{issue_key}.json"))
    _write_json(output_path, payload)
    return payload, output_path, token_usage


def run_agent1(issue_key_hint: str = "") -> tuple[dict[str, Any], Path]:
    source = JiraSource.from_env()
    selected_key = issue_key_hint.strip()
    if selected_key:
        selected_event = source.fetch_ticket_by_key(selected_key)
    else:
        events = source.fetch_new_events(last_seen=None)
        if not events:
            raise RuntimeError("No Jira tickets found for the configured project.")
        selected_event = max(events, key=lambda x: x.created_at)
    payload = {
        "issue_key": selected_event.ticket_key,
        "summary": selected_event.summary,
    }
    output_path = Path(f"agent1/output/{_safe_issue_key(selected_event.ticket_key)}.json")
    _write_json(output_path, payload)
    return payload, output_path


def _save_final_output(issue_key: str, payload: dict[str, Any]) -> Path:
    output_path = Path(f"framework/output/{_safe_issue_key(issue_key)}.json")
    _write_json(output_path, payload)
    return output_path


def run_agent3(ticket_payload: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    issue_key = _safe_issue_key(
        str(ticket_payload.get("Issue key", "")).strip()
        or str(ticket_payload.get("issue_key", "")).strip()
    )
    analyzer = Agent3Analyzer.from_env()
    result = analyzer.analyze_ticket(ticket_payload)
    output_path = Path(f"agent3/output/{issue_key}.json")
    _write_json(output_path, result)
    return result, output_path


def run_agent4(ticket_payload: dict[str, Any], agent3_output: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    issue_key = _safe_issue_key(
        str(ticket_payload.get("Issue key", "")).strip()
        or str(ticket_payload.get("issue_key", "")).strip()
    )
    synthesizer = Agent4RcaSynthesizer.from_env()
    detailed = synthesizer.synthesize(ticket=ticket_payload, agent3_output=agent3_output)
    result = {
        "agent": "agent4",
        "input_ticket": ticket_payload,
        "input_agent3_summary": {
            "similar_tickets_count": len(agent3_output.get("similar_tickets_top3", []))
            if isinstance(agent3_output, dict)
            else 0,
            "documentation_match_count": len(agent3_output.get("documentation_related", []))
            if isinstance(agent3_output, dict)
            else 0,
        },
        "detailed_rca_and_fix": detailed,
    }
    output_path = Path(f"agent4/output/{issue_key}.json")
    _write_json(output_path, result)
    return result, output_path


def run_agent5(agent4_output: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    ticket = agent4_output.get("input_ticket", {}) if isinstance(agent4_output, dict) else {}
    issue_key = _safe_issue_key(
        str(ticket.get("Issue key", "")).strip()
        or str(ticket.get("issue_key", "")).strip()
    )
    analyzer = Agent5RiskAnalyzer.from_env()
    detailed = analyzer.analyze(agent4_output=agent4_output)
    result = {
        "agent": "agent5",
        "input_agent4_summary": {
            "has_rca": bool(agent4_output.get("detailed_rca_and_fix")),
            "ticket_key": str(ticket.get("Issue key", "")),
        },
        "detailed_risk_analysis": detailed,
    }
    output_path = Path(f"agent5/output/{issue_key}.json")
    _write_json(output_path, result)
    return result, output_path


def _ensure_state() -> None:
    defaults = {
        "agent1_output": None,
        "agent2_output": None,
        "agent3_ticket_payload": None,
        "agent3_output": None,
        "agent4_output": None,
        "agent5_output": None,
        "final_output": None,
        "final_output_path": "",
        "agent1_output_path": "",
        "agent2_output_path": "",
        "agent3_output_path": "",
        "agent4_output_path": "",
        "agent5_output_path": "",
        "framework_logs": [],
        "token_events": [],
        "qa_history": [],
        "active_chat_file_path": "",
        "pending_issue_key_input": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _load_existing_outputs(issue_key: str, force: bool = False) -> None:
    safe_key = _safe_issue_key(issue_key)
    candidates = {
        "agent1_output": Path(f"agent1/output/{safe_key}.json"),
        "agent2_output": Path(f"agent2/output/{safe_key}.json"),
        "agent3_output": Path(f"agent3/output/{safe_key}.json"),
        "agent4_output": Path(f"agent4/output/{safe_key}.json"),
        "agent5_output": Path(f"agent5/output/{safe_key}.json"),
    }
    for key, path in candidates.items():
        if not force and st.session_state.get(key) is not None:
            continue
        value = _read_json_if_exists(path)
        if value is not None:
            st.session_state[key] = value
            st.session_state[f"{key}_path"] = str(path.resolve())
            _append_log(f"Loaded existing {key} from {path}")

    if st.session_state.get("agent2_output") and st.session_state.get("agent3_ticket_payload") is None:
        st.session_state["agent3_ticket_payload"] = _map_agent2_to_agent3_ticket(st.session_state["agent2_output"])

    final_path = Path(f"framework/output/{safe_key}.json")
    final_value = _read_json_if_exists(final_path)
    if final_value is not None:
        st.session_state["final_output"] = final_value
        st.session_state["final_output_path"] = str(final_path.resolve())
        _append_log(f"Loaded existing final output from {final_path}")


def _load_agent1_output_for_issue(issue_key: str) -> dict[str, Any] | None:
    safe_key = _safe_issue_key(issue_key)
    path = Path(f"agent1/output/{safe_key}.json")
    value = _read_json_if_exists(path)
    if isinstance(value, dict):
        return value
    return None


def _compose_final_output(issue_key: str) -> dict[str, Any]:
    return {
        "issue_key": issue_key,
        "root_cause_analysis": (
            st.session_state["agent4_output"].get("detailed_rca_and_fix", {}) if st.session_state["agent4_output"] else {}
        ),
        "risk_analysis": (
            st.session_state["agent5_output"].get("detailed_risk_analysis", {})
            if st.session_state["agent5_output"]
            else {}
        ),
        "pipeline_output_paths": {
            "agent2": st.session_state.get("agent2_output_path", ""),
            "agent3": st.session_state.get("agent3_output_path", ""),
            "agent4": st.session_state.get("agent4_output_path", ""),
            "agent5": st.session_state.get("agent5_output_path", ""),
        },
    }


def _short_text(value: str, max_len: int = 220) -> str:
    text = (value or "").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def _clean_jira_text(value: str) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    # Remove Jira wiki mentions, attachment embeds, and image macros.
    text = re.sub(r"\[~accountid:[^\]]+\]", "", text)
    text = re.sub(r"\[\^[^\]]+\]", "", text)
    text = re.sub(r"![^!\n]+!", "", text)
    # Collapse extra whitespace.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _colored_label(text: str, color: str = "#7cc7ff") -> str:
    return f"<span style='color:{color}; font-weight:700'>{text}</span>"


def _format_comment_for_display(comment: str) -> str:
    raw = _clean_jira_text(comment)
    if not raw:
        return ""
    # Legacy format from historical JSON: date;author;message
    parts = raw.split(";", 2)
    if len(parts) == 3:
        date_part = parts[0].strip()
        author_part = parts[1].strip()
        body_part = parts[2].strip()
        if body_part:
            return f"{author_part} ({date_part}): {body_part}"
    return raw


def _safe_name_from_path(path_value: str) -> str:
    try:
        return Path(path_value).name if path_value else ""
    except Exception:
        return ""


def _format_size_bytes(size_value: Any) -> str:
    try:
        size = int(size_value)
    except Exception:
        return "unknown size"
    if size < 1024:
        return f"{size} B"
    if size < (1024 * 1024):
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.2f} MB"


def _render_agent3_attachment_group(title: str, items: list[dict[str, Any]]) -> None:
    st.markdown(_colored_label(title), unsafe_allow_html=True)
    if not items:
        st.caption("None")
        return
    for idx, item in enumerate(items, start=1):
        path_value = str(item.get("path", "")).strip()
        name = _safe_name_from_path(path_value) or f"attachment_{idx}"
        one_line = (
            f"{name} - {str(item.get('type', 'attachment')).strip() or 'attachment'}; "
            f"{_format_size_bytes(item.get('size_bytes'))}. "
            f"{_short_text(str(item.get('note', '')).strip(), 140)}"
        ).strip()
        st.markdown(f"**Attachment {idx} Findings:** {one_line}")
        with st.expander(f"Attachment {idx} Details: {name}", expanded=False):
            if path_value:
                st.caption(f"Path: `{path_value}`")
            st.write(f"Type: {item.get('type', 'N/A')}")
            st.write(f"Exists: {item.get('exists', 'N/A')}")
            st.write(f"Size: {_format_size_bytes(item.get('size_bytes'))}")
            note = str(item.get("note", "")).strip()
            if note:
                st.write(f"Notes: {note}")
            excerpt = str(item.get("excerpt", "")).strip()
            if excerpt:
                st.markdown("**Excerpt**")
                st.code(excerpt)


def _render_final_output_pretty(payload: dict[str, Any]) -> None:
    rca = payload.get("root_cause_analysis", {}) if isinstance(payload, dict) else {}
    risk = payload.get("risk_analysis", {}) if isinstance(payload, dict) else {}
    issue_key = str(payload.get("issue_key", "")).strip()

    if issue_key:
        st.markdown(f"{_colored_label('Ticket:')} <code>{issue_key}</code>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Recommended Fix", str(rca.get("recommended_fix_path", "N/A")))
    c2.metric("Overall Risk", str(risk.get("overall_risk_level", "N/A")))
    c3.metric("Confidence", str(risk.get("confidence", rca.get("confidence", "N/A"))))

    st.markdown(_colored_label("RCA Summary"), unsafe_allow_html=True)
    st.write(str(rca.get("executive_summary", "N/A")))

    st.markdown(_colored_label("Root Cause Analysis"), unsafe_allow_html=True)
    st.write(str(rca.get("root_cause_analysis", "N/A")))

    recommended_fix = str(rca.get("recommended_fix_path", "")).strip()
    with st.expander("Recommended Fix", expanded=True):
        st.write(recommended_fix or "N/A")

    root_cause_factors = rca.get("root_cause_factors", [])
    if isinstance(root_cause_factors, list) and root_cause_factors:
        with st.expander("Root Cause Factors", expanded=False):
            for item in root_cause_factors:
                st.markdown(f"- {item}")

    evidence_used = rca.get("evidence_used", [])
    if isinstance(evidence_used, list) and evidence_used:
        with st.expander("Evidence Used", expanded=False):
            for item in evidence_used:
                st.markdown(f"- {item}")

    possible_fixes = rca.get("possible_fixes", [])
    if isinstance(possible_fixes, list) and possible_fixes:
        st.markdown(_colored_label("Possible Fix Options"), unsafe_allow_html=True)
        for idx, fix in enumerate(possible_fixes, start=1):
            if not isinstance(fix, dict):
                continue
            title = str(fix.get("title", f"Fix {idx}")).strip() or f"Fix {idx}"
            risk_level = str(fix.get("risk_level", "N/A")).strip()
            st.markdown(f"**Fix {idx}:** {title} (Risk: {risk_level})")
            with st.expander(f"Fix {idx} Details", expanded=False):
                st.write(f"Rationale: {fix.get('rationale', 'N/A')}")
                steps = fix.get("verification_steps", [])
                if isinstance(steps, list) and steps:
                    st.markdown("Verification Steps")
                    for step in steps:
                        st.markdown(f"- {step}")

    st.markdown(_colored_label("Risk Summary"), unsafe_allow_html=True)
    st.write(str(risk.get("executive_summary", "N/A")))
    st.write(f"Go/No-Go Recommendation: {risk.get('go_no_go_recommendation', 'N/A')}")

    risk_drivers = risk.get("risk_drivers", [])
    if isinstance(risk_drivers, list) and risk_drivers:
        with st.expander("Risk Drivers", expanded=False):
            for item in risk_drivers:
                st.markdown(f"- {item}")

    prechecks = risk.get("mitigation_prechecks", [])
    if isinstance(prechecks, list) and prechecks:
        with st.expander("Mitigation Prechecks", expanded=False):
            for item in prechecks:
                st.markdown(f"- {item}")

    assumptions = risk.get("assumptions_and_unknowns", [])
    if isinstance(assumptions, list) and assumptions:
        with st.expander("Assumptions / Unknowns", expanded=False):
            for item in assumptions:
                st.markdown(f"- {item}")


def _render_pretty_output(key_prefix: str, payload: dict[str, Any]) -> None:
    if key_prefix == "agent1_output":
        st.markdown(
            f"{_colored_label('Issue Key:')} <code>{payload.get('issue_key', '')}</code>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"{_colored_label('Summary:')} {_short_text(str(payload.get('summary', '')))}",
            unsafe_allow_html=True,
        )
        return

    if key_prefix == "agent2_output":
        c1, c2, c3 = st.columns(3)
        missing = payload.get("Missing Information", [])
        attachments = payload.get("Attachment Paths", [])
        description = _clean_jira_text(str(payload.get("Description", "")).strip())
        organizations = payload.get("Organizations", [])
        comments = payload.get("Comments", [])
        c1.metric("Issue Key", str(payload.get("Issue key", "N/A")))
        c2.metric("Missing Info Items", len(missing) if isinstance(missing, list) else 0)
        c3.metric("Attachments", len(attachments) if isinstance(attachments, list) else 0)
        st.markdown(
            f"{_colored_label('Issue Key:')} <code>{payload.get('Issue key', '')}</code>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"{_colored_label('Summary:')} {_short_text(str(payload.get('Summary', '')))}",
            unsafe_allow_html=True,
        )
        with st.expander("Description", expanded=False):
            st.write(description or "N/A")
        if isinstance(organizations, list) and organizations:
            st.markdown(
                f"{_colored_label('Organizations:')} {', '.join([str(x) for x in organizations])}",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(f"{_colored_label('Organizations:')} N/A", unsafe_allow_html=True)
        if isinstance(comments, list) and comments:
            st.markdown(_colored_label("Comments:"), unsafe_allow_html=True)
            with st.expander(f"Show {len(comments)} comment(s)", expanded=False):
                for idx, c in enumerate(comments[:8], start=1):
                    cleaned = _format_comment_for_display(str(c))
                    st.write(f"{idx}. {_short_text(cleaned, 320)}")
                if len(comments) > 8:
                    st.caption(f"...and {len(comments) - 8} more comment(s).")
        else:
            st.markdown(f"{_colored_label('Comments:')} N/A", unsafe_allow_html=True)
        st.markdown(
            f"{_colored_label('Attachment Status:')} {payload.get('Attachment Status Message', '')}",
            unsafe_allow_html=True,
        )
        if isinstance(missing, list) and missing:
            st.markdown(
                f"{_colored_label('Missing Information:')} {', '.join([str(x) for x in missing])}",
                unsafe_allow_html=True,
            )
        return

    if key_prefix == "agent3_output":
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Similar Tickets", len(payload.get("similar_tickets", []) or payload.get("similar_tickets_top3", [])))
        c2.metric("Documentation Hits", len(payload.get("documentation", []) or payload.get("documentation_related", [])))
        c3.metric("Log Attachments", len(payload.get("logs", [])))
        c4.metric("YML Files", len(payload.get("yml_files", [])))
        analysis = payload.get("detailed_analysis", {})
        st.markdown(
            f"{_colored_label('Executive Summary:')} {_short_text(str(analysis.get('executive_summary', '')), 320)}",
            unsafe_allow_html=True,
        )
        similar = payload.get("similar_tickets_top3", []) or payload.get("similar_tickets", [])
        st.markdown(_colored_label("Similar Tickets"), unsafe_allow_html=True)
        if isinstance(similar, list) and similar:
            for idx, row in enumerate(similar, start=1):
                if not isinstance(row, dict):
                    continue
                issue_key = str(row.get("issue_key", f"TICKET-{idx}")).strip()
                score = row.get("score", "N/A")
                summary = _short_text(str(row.get("summary", "")).strip(), 120)
                st.markdown(f"**{issue_key}** (Score: {score}) - {summary}")
                with st.expander(f"{issue_key} details", expanded=False):
                    ai_summary = str(row.get("ai_summary", "")).strip()
                    if ai_summary:
                        st.markdown("AI Summary")
                        st.write(ai_summary)
                    desc = str(row.get("description", "")).strip()
                    if desc:
                        st.markdown("Description")
                        st.write(desc)
                    st.write(f"Organizations: {row.get('organizations', 'N/A')}")
                    st.write(f"Updated: {row.get('updated', 'N/A')}")
                    st.write(f"Matched Vector: {row.get('matched_vector_id', 'N/A')}")
        else:
            st.caption("No similar tickets found.")

        docs = payload.get("documentation_related", []) or payload.get("documentation", [])
        if isinstance(docs, list) and docs:
            with st.expander("Documentation Hits", expanded=False):
                for idx, doc in enumerate(docs, start=1):
                    if not isinstance(doc, dict):
                        continue
                    path_value = str(doc.get("path", "")).strip()
                    path_name = _safe_name_from_path(path_value) or path_value or f"doc_{idx}"
                    st.markdown(
                        f"**{idx}.** {path_name} (Score: {doc.get('score', 'N/A')}, "
                        f"Chunk: {doc.get('chunk_index', 'N/A')}/{doc.get('chunk_count', 'N/A')})"
                    )

        _render_agent3_attachment_group("Attachment Analysis: Logs", payload.get("logs", []))
        _render_agent3_attachment_group("Attachment Analysis: YML Files", payload.get("yml_files", []))
        _render_agent3_attachment_group("Attachment Analysis: Other Attachments", payload.get("other_attachments", []))
        return

    if key_prefix == "agent4_output":
        detail = payload.get("detailed_rca_and_fix", {})
        st.markdown(
            f"{_colored_label('RCA Summary:')} {_short_text(str(detail.get('executive_summary', '')), 320)}",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"{_colored_label('Root Cause:')} {_short_text(str(detail.get('root_cause_analysis', '')), 320)}",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"{_colored_label('Recommended Fix Path:')} {detail.get('recommended_fix_path', '')}",
            unsafe_allow_html=True,
        )
        return

    if key_prefix == "agent5_output":
        detail = payload.get("detailed_risk_analysis", {})
        c1, c2 = st.columns(2)
        c1.metric("Overall Risk", str(detail.get("overall_risk_level", "N/A")))
        c2.metric("Confidence", str(detail.get("confidence", "N/A")))
        st.markdown(
            f"{_colored_label('Executive Summary:')} {_short_text(str(detail.get('executive_summary', '')), 320)}",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"{_colored_label('Go/No-Go:')} {detail.get('go_no_go_recommendation', '')}",
            unsafe_allow_html=True,
        )
        return


def _render_output_block(title: str, payload: dict[str, Any] | None, path_text: str, key_prefix: str) -> None:
    with st.expander(title, expanded=False):
        if payload is None:
            st.caption("No output yet.")
            return
        _render_pretty_output(key_prefix, payload)
        if path_text:
            st.caption(f"Saved at: `{path_text}`")


def _run_full_pipeline(issue_key: str) -> None:
    _append_log(f"Starting full pipeline for {issue_key}")
    st.session_state["active_chat_file_path"] = ""
    agent2_output, agent2_path, token_usage = run_agent2(issue_key)
    st.session_state["agent2_output"] = agent2_output
    st.session_state["agent2_output_path"] = str(agent2_path.resolve())
    _record_token_event("agent2", "categorization", token_usage.get("categorization", {}))
    _record_token_event("agent2", "secondary_enrichment", token_usage.get("secondary_enrichment", {}))
    _append_log("Agent2 completed")

    ticket_payload = _map_agent2_to_agent3_ticket(agent2_output)
    st.session_state["agent3_ticket_payload"] = ticket_payload
    agent3_output, agent3_path = run_agent3(ticket_payload)
    st.session_state["agent3_output"] = agent3_output
    st.session_state["agent3_output_path"] = str(agent3_path.resolve())
    _record_agent_usage("agent3", agent3_output)
    _append_log("Agent3 completed")

    agent4_output, agent4_path = run_agent4(ticket_payload, agent3_output)
    st.session_state["agent4_output"] = agent4_output
    st.session_state["agent4_output_path"] = str(agent4_path.resolve())
    _record_agent_usage("agent4", agent4_output)
    _append_log("Agent4 completed")

    agent5_output, agent5_path = run_agent5(agent4_output)
    st.session_state["agent5_output"] = agent5_output
    st.session_state["agent5_output_path"] = str(agent5_path.resolve())
    _record_agent_usage("agent5", agent5_output)
    _append_log("Agent5 completed")

    st.session_state["final_output"] = _compose_final_output(issue_key)
    final_path = _save_final_output(issue_key, st.session_state["final_output"])
    st.session_state["final_output_path"] = str(final_path.resolve())
    _append_log("Full pipeline completed")


def _answer_followup(issue_key: str, final_output: dict[str, Any], question: str) -> tuple[str, dict[str, int]]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return (
            "OPENAI_API_KEY is missing. Add it in .env to enable clarification Q&A.",
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )
    client = OpenAI(api_key=api_key)
    model = os.getenv("AGENT_UI_QA_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    try:
        response = client.responses.create(
            model=model,
            temperature=0.2,
            max_output_tokens=800,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You answer follow-up questions about an existing RCA and risk analysis. "
                        "Use only the provided context. If context is insufficient, say exactly what is missing."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "ticket_key": issue_key,
                            "final_output": final_output,
                            "question": question,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        )
        usage = _normalize_usage_dict(getattr(response, "usage", None))
        answer = (getattr(response, "output_text", "") or "").strip()
        if not answer:
            answer = "No answer generated."
        return answer, usage
    except Exception as exc:
        return (
            f"Clarification request failed: {exc}",
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )


def main() -> None:
    st.set_page_config(page_title="RCA Agentic Framework", layout="wide")
    load_env_file(".env")
    _ensure_state()
    if st.session_state.get("pending_issue_key_input"):
        st.session_state["issue_key_input"] = st.session_state["pending_issue_key_input"]
        st.session_state["pending_issue_key_input"] = ""

    st.title("RCA Agentic Framework")
    st.caption("Manual test mode: run agents individually or run full pipeline for a Jira ticket key.")

    with st.sidebar:
        st.subheader("Run Controls")
        issue_key = st.text_input("Jira Ticket Key", key="issue_key_input", value="JMCH-1802").strip()
        load_saved = st.checkbox("Auto-load saved output files for this ticket", value=False)
        st.markdown("---")
        st.subheader("Previous Analysis")
        previous_keys = _discover_previous_issue_keys()
        selected_previous = st.selectbox(
            "Select Previous Ticket",
            options=[""] + previous_keys,
            index=0,
            help="Load previously saved agent outputs/final output for a past run.",
        )
        if st.button("Fetch Previous Analysis", use_container_width=True):
            target_key = (selected_previous or issue_key).strip()
            if not target_key:
                st.warning("Provide or select a ticket key first.")
            else:
                _load_existing_outputs(target_key, force=True)
                _queue_issue_key_update(target_key)
                if st.session_state.get("agent4_output") and st.session_state.get("agent5_output"):
                    st.session_state["final_output"] = _compose_final_output(target_key)
                    final_path = _save_final_output(target_key, st.session_state["final_output"])
                    st.session_state["final_output_path"] = str(final_path.resolve())
                _append_log(f"Fetched previous analysis for {target_key}")
                st.success(f"Loaded previous analysis for {target_key}")
                st.rerun()
        st.markdown("---")
        st.subheader("Previous Chats")
        chat_filter_key = st.text_input(
            "Chat Ticket Filter (optional)",
            value=issue_key,
            help="Show chats for this ticket key. Clear to show all chats.",
        ).strip()
        chat_files = _discover_chat_files(chat_filter_key or None)
        chat_options = [""] + [str(p) for p in chat_files]
        selected_chat_file = st.selectbox("Select Chat Session", options=chat_options, index=0)
        if st.button("Load Chat Session", use_container_width=True):
            if not selected_chat_file:
                st.warning("Select a chat session first.")
            else:
                chat_payload = _load_chat_session(Path(selected_chat_file))
                loaded_ticket_key = str(chat_payload.get("ticket_key", "")).strip()
                loaded_final_output = chat_payload.get("final_output", {})
                loaded_qa_history = chat_payload.get("qa_history", [])
                if loaded_ticket_key:
                    _queue_issue_key_update(loaded_ticket_key)
                st.session_state["final_output"] = loaded_final_output if isinstance(loaded_final_output, dict) else None
                st.session_state["qa_history"] = loaded_qa_history if isinstance(loaded_qa_history, list) else []
                st.session_state["active_chat_file_path"] = str(Path(selected_chat_file).resolve())
                _append_log(f"Loaded chat session: {selected_chat_file}")
                st.success(f"Loaded chat session: {Path(selected_chat_file).name}")
                st.rerun()
        if st.button("Reset Session State", use_container_width=True):
            for key in [
                "agent1_output",
                "agent2_output",
                "agent3_ticket_payload",
                "agent3_output",
                "agent4_output",
                "agent5_output",
                "final_output",
                "final_output_path",
                "agent1_output_path",
                "agent2_output_path",
                "agent3_output_path",
                "agent4_output_path",
                "agent5_output_path",
                "framework_logs",
                "token_events",
                "qa_history",
                "active_chat_file_path",
            ]:
                st.session_state[key] = [] if key in {"framework_logs", "token_events", "qa_history"} else ("" if key.endswith("_path") else None)
            _queue_issue_key_update("JMCH-1802")
            st.rerun()

    if not issue_key:
        st.warning("Enter a Jira ticket key to continue.")
        return

    if load_saved:
        _load_existing_outputs(issue_key)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    run_full = c1.button("Full Pipeline", use_container_width=True, type="primary")
    run_a1 = c2.button("Ticket Ingestion", use_container_width=True)
    run_a2 = c3.button("Validation & Enrichment", use_container_width=True)
    run_a3 = c4.button("Multi-Source Analysis", use_container_width=True)
    run_a4 = c5.button("RCA Synthesis", use_container_width=True)
    run_a5 = c6.button("Risk & Reporting", use_container_width=True)

    try:
        if run_a1:
            with st.spinner("Running Agent1..."):
                agent1_output, agent1_path = run_agent1(issue_key)
                st.session_state["agent1_output"] = agent1_output
                st.session_state["agent1_output_path"] = str(agent1_path.resolve())
                # Running Agent1 should not trigger or keep downstream outputs.
                st.session_state["agent2_output"] = None
                st.session_state["agent3_ticket_payload"] = None
                st.session_state["agent3_output"] = None
                st.session_state["agent4_output"] = None
                st.session_state["agent5_output"] = None
                st.session_state["final_output"] = None
                st.session_state["agent2_output_path"] = ""
                st.session_state["agent3_output_path"] = ""
                st.session_state["agent4_output_path"] = ""
                st.session_state["agent5_output_path"] = ""
                st.session_state["final_output_path"] = ""
                detected_key = str(agent1_output.get("issue_key", "")).strip()
                if detected_key:
                    _queue_issue_key_update(detected_key)
                _append_log(f"Agent1 completed. Ticket selected: {detected_key or 'N/A'}")
            st.success(f"Agent1 completed. Using ticket key: {detected_key or issue_key}")
            st.rerun()

        if run_full:
            with st.spinner("Running Agent2 -> Agent3 -> Agent4 -> Agent5..."):
                _run_full_pipeline(st.session_state.get("issue_key_input", issue_key))
            st.success("Full framework run completed.")

        if run_a2:
            agent1_output = st.session_state.get("agent1_output")
            if not isinstance(agent1_output, dict) or not str(agent1_output.get("issue_key", "")).strip():
                from_file = _load_agent1_output_for_issue(issue_key)
                if isinstance(from_file, dict):
                    st.session_state["agent1_output"] = from_file
                    st.session_state["agent1_output_path"] = str(
                        Path(f"agent1/output/{_safe_issue_key(issue_key)}.json").resolve()
                    )
                    agent1_output = from_file
                    _append_log(f"Loaded Agent1 output from file for {issue_key}")
            if not isinstance(agent1_output, dict) or not str(agent1_output.get("issue_key", "")).strip():
                raise RuntimeError("Agent2 requires Agent1 output. Run Agent1 first for input.")
            issue_key_from_agent1 = str(agent1_output.get("issue_key", "")).strip()
            with st.spinner("Running Agent2..."):
                agent2_output, agent2_path, token_usage = run_agent2(issue_key_from_agent1)
                st.session_state["agent2_output"] = agent2_output
                st.session_state["agent2_output_path"] = str(agent2_path.resolve())
                st.session_state["agent3_ticket_payload"] = _map_agent2_to_agent3_ticket(agent2_output)
                _record_token_event("agent2", "categorization", token_usage.get("categorization", {}))
                _record_token_event("agent2", "secondary_enrichment", token_usage.get("secondary_enrichment", {}))
                _append_log("Agent2 completed")
            st.success("Agent2 completed.")

        if run_a3:
            ticket_payload = st.session_state.get("agent3_ticket_payload")
            if not ticket_payload:
                raise RuntimeError("Agent3 requires ticket payload. Run Agent2 first or load Agent2 output.")
            with st.spinner("Running Agent3..."):
                agent3_output, agent3_path = run_agent3(ticket_payload)
                st.session_state["agent3_output"] = agent3_output
                st.session_state["agent3_output_path"] = str(agent3_path.resolve())
                _record_agent_usage("agent3", agent3_output)
                _append_log("Agent3 completed")
            st.success("Agent3 completed.")

        if run_a4:
            ticket_payload = st.session_state.get("agent3_ticket_payload")
            agent3_output = st.session_state.get("agent3_output")
            if not ticket_payload or not agent3_output:
                raise RuntimeError("Agent4 requires Agent3 ticket payload and Agent3 output. Run Agent3 first.")
            with st.spinner("Running Agent4..."):
                agent4_output, agent4_path = run_agent4(ticket_payload, agent3_output)
                st.session_state["agent4_output"] = agent4_output
                st.session_state["agent4_output_path"] = str(agent4_path.resolve())
                _record_agent_usage("agent4", agent4_output)
                _append_log("Agent4 completed")
            st.success("Agent4 completed.")

        if run_a5:
            agent4_output = st.session_state.get("agent4_output")
            if not agent4_output:
                raise RuntimeError("Agent5 requires Agent4 output. Run Agent4 first.")
            with st.spinner("Running Agent5..."):
                agent5_output, agent5_path = run_agent5(agent4_output)
                st.session_state["agent5_output"] = agent5_output
                st.session_state["agent5_output_path"] = str(agent5_path.resolve())
                _record_agent_usage("agent5", agent5_output)
                _append_log("Agent5 completed")
            st.success("Agent5 completed.")

    except Exception as exc:
        _append_log(f"Error: {exc}")
        st.error(f"Pipeline error: {exc}")
        st.code(traceback.format_exc())

    if st.session_state.get("agent4_output") and st.session_state.get("agent5_output"):
        st.session_state["final_output"] = _compose_final_output(issue_key)

    st.subheader("Framework Final Output")
    final_output = st.session_state.get("final_output")
    with st.expander("Final RCA + Risk Output", expanded=True):
        if final_output is None:
            st.caption("Run full framework or run Agent4 and Agent5 to generate final output.")
        else:
            final_output_path = st.session_state.get("final_output_path", "")
            if final_output_path:
                st.caption(f"Saved at: `{final_output_path}`")
            st.download_button(
                label="Download Final Output JSON",
                data=_json_dump(final_output),
                file_name=f"framework_final_{_safe_issue_key(issue_key)}.json",
                mime="application/json",
                key="dl_final_output",
            )
            _render_final_output_pretty(final_output)

    st.subheader("Clarifications")
    if final_output is None:
        st.caption("Generate or fetch final output first, then ask follow-up questions.")
    else:
        active_chat_file = st.session_state.get("active_chat_file_path", "")
        if active_chat_file:
            st.caption(f"Active chat file: `{active_chat_file}`")
        followup_question = st.text_area(
            "Ask a follow-up question about RCA or risk analysis",
            key="followup_question",
            placeholder="Example: Why is recurrence risk medium and what evidence supports it?",
        )
        if st.button("Ask Clarification", use_container_width=False):
            question = (followup_question or "").strip()
            if not question:
                st.warning("Enter a question before asking for clarification.")
            else:
                with st.spinner("Generating clarification..."):
                    answer, usage = _answer_followup(issue_key, final_output, question)
                    st.session_state["qa_history"].append(
                        {
                            "asked_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                            "question": question,
                            "answer": answer,
                            "token_usage": usage,
                        }
                    )
                    _record_token_event("clarification", "qa_followup", usage)
                    _append_log("Clarification answered")
                    chat_path = _save_chat_session(issue_key, final_output, st.session_state["qa_history"])
                    if chat_path is not None:
                        _append_log(f"Chat saved: {chat_path}")
                st.success("Clarification generated.")

        if st.session_state["qa_history"]:
            for idx, row in enumerate(reversed(st.session_state["qa_history"]), start=1):
                with st.expander(f"Q&A #{idx}", expanded=False):
                    st.caption(f"Asked at (UTC): {row.get('asked_at_utc', '')}")
                    st.markdown(f"**Q:** {row.get('question', '')}")
                    st.markdown(f"**A:** {row.get('answer', '')}")

    st.subheader("Agent Outputs")
    _render_output_block(
        "Agent1 Output",
        st.session_state.get("agent1_output"),
        st.session_state.get("agent1_output_path", ""),
        "agent1_output",
    )
    _render_output_block(
        "Agent2 Output",
        st.session_state.get("agent2_output"),
        st.session_state.get("agent2_output_path", ""),
        "agent2_output",
    )
    _render_output_block(
        "Agent3 Output",
        st.session_state.get("agent3_output"),
        st.session_state.get("agent3_output_path", ""),
        "agent3_output",
    )
    _render_output_block(
        "Agent4 Output",
        st.session_state.get("agent4_output"),
        st.session_state.get("agent4_output_path", ""),
        "agent4_output",
    )
    _render_output_block(
        "Agent5 Output",
        st.session_state.get("agent5_output"),
        st.session_state.get("agent5_output_path", ""),
        "agent5_output",
    )

    st.subheader("Execution Log")
    if st.session_state["framework_logs"]:
        for log_line in st.session_state["framework_logs"][-30:]:
            st.text(f"- {log_line}")
    else:
        st.caption("No events yet.")


if __name__ == "__main__":
    main()
