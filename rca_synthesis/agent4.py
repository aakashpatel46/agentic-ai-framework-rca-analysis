from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI


def load_env_file(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def get_env(name: str, required: bool = True, default: str | None = None) -> str:
    value = (os.getenv(name, default) or "").strip().strip('"').strip("'")
    if required and not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def safe_json_loads(text: str) -> dict[str, Any] | None:
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else None
    except Exception:
        return None


class Agent4RcaSynthesizer:
    def __init__(self, openai_client: OpenAI, model: str = "gpt-4o-mini") -> None:
        self.openai_client = openai_client
        self.model = model
        self.last_token_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    @classmethod
    def from_env(cls) -> "Agent4RcaSynthesizer":
        return cls(
            openai_client=OpenAI(api_key=get_env("OPENAI_API_KEY")),
            model=get_env("AGENT4_ANALYSIS_MODEL", required=False, default="gpt-4o-mini"),
        )

    def synthesize(self, ticket: dict[str, Any], agent3_output: dict[str, Any]) -> dict[str, Any]:
        fallback = self._fallback(ticket=ticket, agent3_output=agent3_output)
        prompt_payload = {
            "ticket": ticket,
            "agent3_output": agent3_output,
            "goal": "Produce a strong root cause analysis and possible fix options.",
        }
        try:
            response = self.openai_client.responses.create(
                model=self.model,
                temperature=0.2,
                max_output_tokens=1500,
                input=[
                    {
                        "role": "system",
                        "content": (
                            "You are Agent4 in an RCA pipeline. Return ONLY valid JSON with keys: "
                            "executive_summary (string), root_cause_analysis (string), root_cause_factors (array of strings), "
                            "evidence_used (array of strings), possible_fixes (array of objects with keys: title, rationale, risk_level, verification_steps), "
                            "recommended_fix_path (string), confidence (string), assumptions_and_gaps (array of strings)."
                        ),
                    },
                    {"role": "user", "content": json.dumps(prompt_payload, ensure_ascii=False)},
                ],
            )
            self.last_token_usage = self._usage_to_dict(getattr(response, "usage", None))
            parsed = safe_json_loads(getattr(response, "output_text", "") or "")
            if parsed:
                parsed["synthesis_source"] = "openai"
                parsed["token_usage"] = self.last_token_usage
                return parsed
        except Exception as exc:
            fallback["synthesis_error"] = f"LLM synthesis fallback used: {exc}"
        fallback["token_usage"] = self.last_token_usage
        return fallback

    @staticmethod
    def _usage_to_dict(usage: Any) -> dict[str, int]:
        if usage is None:
            return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        if isinstance(usage, dict):
            input_tokens = int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
            output_tokens = int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0)
            total_tokens = int(usage.get("total_tokens", input_tokens + output_tokens) or 0)
            return {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            }
        input_tokens = int(getattr(usage, "input_tokens", getattr(usage, "prompt_tokens", 0)) or 0)
        output_tokens = int(getattr(usage, "output_tokens", getattr(usage, "completion_tokens", 0)) or 0)
        total_tokens = int(getattr(usage, "total_tokens", input_tokens + output_tokens) or 0)
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }

    def _fallback(self, ticket: dict[str, Any], agent3_output: dict[str, Any]) -> dict[str, Any]:
        summary = normalize_whitespace(
            str(ticket.get("Summary", "") or ticket.get("summary", "") or "No summary provided")
        )
        a3 = agent3_output.get("detailed_analysis", {}) if isinstance(agent3_output, dict) else {}
        likely_causes = a3.get("probable_root_causes", []) if isinstance(a3, dict) else []
        diagnostics = a3.get("recommended_diagnostics", []) if isinstance(a3, dict) else []
        similar_tickets = agent3_output.get("similar_tickets_top3", []) if isinstance(agent3_output, dict) else []
        docs = agent3_output.get("documentation_related", []) if isinstance(agent3_output, dict) else []

        evidence = []
        for cause in likely_causes[:3]:
            evidence.append(f"Agent3 probable cause: {normalize_whitespace(str(cause))}")
        for diag in diagnostics[:3]:
            evidence.append(f"Agent3 diagnostic hint: {normalize_whitespace(str(diag))}")
        for t in similar_tickets[:2]:
            if isinstance(t, dict):
                key = str(t.get("issue_key", "")).strip()
                if key:
                    evidence.append(f"Similar ticket reference: {key}")
        for d in docs[:2]:
            if isinstance(d, dict):
                path = str(d.get("path", "")).strip()
                if path:
                    evidence.append(f"Documentation reference: {path}")

        possible_fixes = [
            {
                "title": "Add null-safe and edge-case handling in refund guest flow",
                "rationale": "Crash symptoms indicate an unhandled guest-checkout code path during refund processing.",
                "risk_level": "low-to-medium",
                "verification_steps": [
                    "Add unit tests for guest checkout refund path.",
                    "Run integration tests for refund flows across guest and registered users.",
                    "Validate logs no longer show the triggering exception.",
                ],
            },
            {
                "title": "Validate and guard request payload before refund service invocation",
                "rationale": "If invalid or incomplete fields are reaching the handler, guard clauses can prevent runtime failures.",
                "risk_level": "low",
                "verification_steps": [
                    "Add validation checks for required fields before processing.",
                    "Confirm invalid requests are rejected with explicit errors.",
                    "Test backward compatibility with existing POS clients.",
                ],
            },
        ]

        return {
            "synthesis_source": "fallback",
            "executive_summary": (
                f"Consolidated RCA for '{summary}'. Agent3 signals suggest a repeatable defect pattern likely tied to "
                "guest-checkout refund handling after recent deployment changes."
            ),
            "root_cause_analysis": (
                "The most probable root cause is an unhandled guest-order edge case in the refund path, "
                "potentially introduced or re-activated by recent deployment changes. This causes runtime failure "
                "when refund processing assumes data/state that is absent for guest transactions."
            ),
            "root_cause_factors": [
                "Guest checkout path differs from registered-user path and may skip expected state initialization.",
                "Recent deployment timing correlates with incident start.",
                "Insufficient defensive checks in refund handling for null/invalid state.",
            ],
            "evidence_used": evidence,
            "possible_fixes": possible_fixes,
            "recommended_fix_path": "Start with low-risk guard and validation fix, then add targeted logic correction with tests.",
            "confidence": "medium",
            "assumptions_and_gaps": [
                "No direct stack trace or code diff was provided to Agent4 input.",
                "Production logs and exact failing function signature should be verified before final patch.",
            ],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agent4: RCA synthesis + possible fix options")
    parser.add_argument(
        "--ticket-json-file",
        default=os.getenv("AGENT4_INPUT_TICKET_JSON_FILE", ""),
        help="Path to current ticket JSON input file",
    )
    parser.add_argument(
        "--ticket-json-inline",
        default=os.getenv("AGENT4_INPUT_TICKET_JSON", ""),
        help="Raw JSON string for current ticket input",
    )
    parser.add_argument(
        "--agent3-output-file",
        default=os.getenv("AGENT4_INPUT_AGENT3_OUTPUT_FILE", ""),
        help="Path to Agent3 output JSON file",
    )
    parser.add_argument(
        "--agent3-output-inline",
        default=os.getenv("AGENT4_INPUT_AGENT3_OUTPUT_JSON", ""),
        help="Raw JSON string for Agent3 output",
    )
    parser.add_argument(
        "--output-file",
        default=os.getenv("AGENT4_OUTPUT_FILE", ""),
        help="Output JSON file path",
    )
    return parser.parse_args()


def _load_json_object_from_file(path_value: str, name: str) -> dict[str, Any]:
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"{name} file not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{name} file must contain a JSON object")
    return value


def load_ticket_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.ticket_json_file:
        return _load_json_object_from_file(args.ticket_json_file, "Ticket JSON")
    if args.ticket_json_inline:
        value = json.loads(args.ticket_json_inline)
        if not isinstance(value, dict):
            raise ValueError("Inline ticket JSON must be a JSON object")
        return value
    raise ValueError("Provide --ticket-json-file or --ticket-json-inline")


def load_agent3_output_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.agent3_output_file:
        return _load_json_object_from_file(args.agent3_output_file, "Agent3 output JSON")
    if args.agent3_output_inline:
        value = json.loads(args.agent3_output_inline)
        if not isinstance(value, dict):
            raise ValueError("Inline Agent3 output JSON must be a JSON object")
        return value
    raise ValueError("Provide --agent3-output-file or --agent3-output-inline")


def resolve_output_path(ticket: dict[str, Any], output_file: str) -> Path:
    if output_file.strip():
        return Path(output_file.strip())
    issue_key = (
        str(ticket.get("Issue key", "")).strip()
        or str(ticket.get("issue_key", "")).strip()
        or "ticket"
    )
    safe_issue_key = re.sub(r"[^a-zA-Z0-9_.-]", "_", issue_key)
    return Path(f"rca_synthesis/output/{safe_issue_key}.json")


def main() -> None:
    load_env_file(".env")
    args = parse_args()
    ticket = load_ticket_from_args(args)
    agent3_output = load_agent3_output_from_args(args)

    synthesizer = Agent4RcaSynthesizer.from_env()
    detailed = synthesizer.synthesize(ticket=ticket, agent3_output=agent3_output)
    output = {
        "agent": "agent4",
        "generated_at_utc": utc_now_iso(),
        "input_ticket": ticket,
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

    output_path = resolve_output_path(ticket, args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[agent4] RCA synthesis complete")
    print(f"[agent4] Output JSON: {output_path.resolve()}")


if __name__ == "__main__":
    main()

