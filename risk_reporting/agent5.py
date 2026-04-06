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


class Agent5RiskAnalyzer:
    def __init__(self, openai_client: OpenAI, model: str = "gpt-4o-mini") -> None:
        self.openai_client = openai_client
        self.model = model
        self.last_token_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    @classmethod
    def from_env(cls) -> "Agent5RiskAnalyzer":
        return cls(
            openai_client=OpenAI(api_key=get_env("OPENAI_API_KEY")),
            model=get_env("AGENT5_ANALYSIS_MODEL", required=False, default="gpt-4o-mini"),
        )

    def analyze(self, agent4_output: dict[str, Any]) -> dict[str, Any]:
        fallback = self._fallback(agent4_output)
        prompt_payload = {
            "agent4_output": agent4_output,
            "constraints": "Use only this input. Do not assume external logs, docs, or telemetry.",
        }
        try:
            response = self.openai_client.responses.create(
                model=self.model,
                temperature=0.2,
                max_output_tokens=1400,
                input=[
                    {
                        "role": "system",
                        "content": (
                            "You are Agent5 in an RCA pipeline. Return ONLY valid JSON with keys: "
                            "executive_summary (string), overall_risk_level (string), "
                            "risk_dimensions (object with keys: blast_radius, regression_risk, data_integrity_risk, deployment_risk, recurrence_risk, sla_impact; "
                            "each must include level and reasoning), "
                            "risk_drivers (array of strings), mitigation_prechecks (array of strings), "
                            "go_no_go_recommendation (string), confidence (string), assumptions_and_unknowns (array of strings). "
                            "Base analysis strictly on provided Agent4 output."
                        ),
                    },
                    {"role": "user", "content": json.dumps(prompt_payload, ensure_ascii=False)},
                ],
            )
            self.last_token_usage = self._usage_to_dict(getattr(response, "usage", None))
            parsed = safe_json_loads(getattr(response, "output_text", "") or "")
            if parsed:
                parsed["analysis_source"] = "openai"
                parsed["token_usage"] = self.last_token_usage
                return parsed
        except Exception as exc:
            fallback["analysis_error"] = f"LLM synthesis fallback used: {exc}"
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

    def _fallback(self, agent4_output: dict[str, Any]) -> dict[str, Any]:
        details = agent4_output.get("detailed_rca_and_fix", {}) if isinstance(agent4_output, dict) else {}
        issue = agent4_output.get("input_ticket", {}) if isinstance(agent4_output, dict) else {}
        summary = normalize_whitespace(
            str(issue.get("Summary", "") or issue.get("summary", "") or "Ticket under analysis")
        )
        fixes = details.get("possible_fixes", []) if isinstance(details, dict) else []
        factors = details.get("root_cause_factors", []) if isinstance(details, dict) else []
        confidence = str(details.get("confidence", "medium")).strip() if isinstance(details, dict) else "medium"

        risk_level = "medium"
        if str(issue.get("Priority", "")).lower() in {"critical", "highest", "p1"}:
            risk_level = "high"

        risk_dimensions = {
            "blast_radius": {
                "level": "medium",
                "reasoning": "Issue is in a core refund path and can impact affected transaction flows if widespread.",
            },
            "regression_risk": {
                "level": "medium",
                "reasoning": "Fixes touch transactional logic; behavior can regress without targeted tests.",
            },
            "data_integrity_risk": {
                "level": "low-to-medium",
                "reasoning": "No explicit schema/data mutation issue stated, but transactional errors may affect refund state consistency.",
            },
            "deployment_risk": {
                "level": "medium",
                "reasoning": "Changes in production-sensitive flow require staged rollout and validation checks.",
            },
            "recurrence_risk": {
                "level": "medium",
                "reasoning": "Edge-case handling gaps can recur if not covered by regression tests and guardrails.",
            },
            "sla_impact": {
                "level": "medium-to-high",
                "reasoning": "Refund-flow instability can threaten response/resolve timelines for customer-facing incidents.",
            },
        }

        drivers = [
            "Core transaction/refund path involvement.",
            "Root cause indicates edge-case handling weakness.",
            "Incident tied to recent deployment timing.",
        ]
        if factors:
            drivers.extend([normalize_whitespace(str(x)) for x in factors[:3]])

        prechecks = [
            "Confirm full regression test coverage for guest and registered user refund scenarios.",
            "Validate monitoring and alert thresholds around refund failure rates before rollout.",
            "Perform canary or phased rollout with rollback criteria defined.",
        ]
        if fixes:
            prechecks.append("Verify proposed fix paths are mutually compatible and sequenced by risk.")

        return {
            "analysis_source": "fallback",
            "executive_summary": f"Risk analysis for '{summary}' based only on Agent4 RCA and fix candidates.",
            "overall_risk_level": risk_level,
            "risk_dimensions": risk_dimensions,
            "risk_drivers": drivers,
            "mitigation_prechecks": prechecks,
            "go_no_go_recommendation": (
                "GO with controlled rollout only after prechecks pass; otherwise NO-GO until regression risk is reduced."
            ),
            "confidence": confidence or "medium",
            "assumptions_and_unknowns": [
                "No live production telemetry or error-volume metrics were provided to Agent5.",
                "No dependency graph or downstream service impact map was provided.",
            ],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agent5: Risk analysis based on Agent4 output")
    parser.add_argument(
        "--agent4-output-file",
        default=os.getenv("AGENT5_INPUT_AGENT4_OUTPUT_FILE", ""),
        help="Path to Agent4 output JSON file",
    )
    parser.add_argument(
        "--agent4-output-inline",
        default=os.getenv("AGENT5_INPUT_AGENT4_OUTPUT_JSON", ""),
        help="Raw JSON string for Agent4 output",
    )
    parser.add_argument(
        "--output-file",
        default=os.getenv("AGENT5_OUTPUT_FILE", ""),
        help="Output JSON file path",
    )
    return parser.parse_args()


def load_agent4_output_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.agent4_output_file:
        path = Path(args.agent4_output_file)
        if not path.exists():
            raise FileNotFoundError(f"Agent4 output file not found: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Agent4 output file must contain a JSON object")
        return value
    if args.agent4_output_inline:
        value = json.loads(args.agent4_output_inline)
        if not isinstance(value, dict):
            raise ValueError("Inline Agent4 output JSON must be a JSON object")
        return value
    raise ValueError("Provide --agent4-output-file or --agent4-output-inline")


def resolve_output_path(agent4_output: dict[str, Any], output_file: str) -> Path:
    if output_file.strip():
        return Path(output_file.strip())
    ticket = agent4_output.get("input_ticket", {}) if isinstance(agent4_output, dict) else {}
    issue_key = (
        str(ticket.get("Issue key", "")).strip()
        or str(ticket.get("issue_key", "")).strip()
        or "ticket"
    )
    safe_issue_key = re.sub(r"[^a-zA-Z0-9_.-]", "_", issue_key)
    return Path(f"risk_reporting/output/{safe_issue_key}.json")


def main() -> None:
    load_env_file(".env")
    args = parse_args()
    agent4_output = load_agent4_output_from_args(args)

    analyzer = Agent5RiskAnalyzer.from_env()
    detailed = analyzer.analyze(agent4_output=agent4_output)
    output = {
        "agent": "agent5",
        "generated_at_utc": utc_now_iso(),
        "input_agent4_summary": {
            "has_rca": bool(agent4_output.get("detailed_rca_and_fix")),
            "ticket_key": str(
                (agent4_output.get("input_ticket", {}) if isinstance(agent4_output, dict) else {}).get("Issue key", "")
            ),
        },
        "detailed_risk_analysis": detailed,
    }

    output_path = resolve_output_path(agent4_output, args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[agent5] Risk analysis complete")
    print(f"[agent5] Output JSON: {output_path.resolve()}")


if __name__ == "__main__":
    main()

