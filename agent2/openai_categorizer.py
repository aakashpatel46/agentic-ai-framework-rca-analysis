from __future__ import annotations

import json
import os
from typing import Any

import requests

from agent2.models import RawTicket
from agent2.rules import RuleSet


class OpenAICategorizer:
    def __init__(self, rules: RuleSet) -> None:
        self._rules = rules
        self._api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self._model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
        self.last_categorize_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        self.last_secondary_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    def categorize(self, ticket: RawTicket) -> str:
        if not self._api_key:
            self.last_categorize_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            return self._fallback_category(ticket)

        categories = self._rules.category_names()
        prompt = self._build_prompt(ticket, categories)

        response = requests.post(
            "https://api.openai.com/v1/responses",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._model,
                "input": prompt,
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "ticket_category",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "category": {
                                    "type": "string",
                                    "enum": categories,
                                }
                            },
                            "required": ["category"],
                            "additionalProperties": False,
                        },
                    }
                },
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        self.last_categorize_usage = self._extract_usage(payload)
        category = self._extract_category(payload)
        if category in categories:
            return category
        return "other"

    def enrich_missing_information(
        self,
        ticket: RawTicket,
        category: str,
        missing_information: list[str],
        secondary_prompt: str,
    ) -> dict[str, Any]:
        input_payload = {
            "raw_ticket": ticket.raw,
            "attachment_names": [att.filename for att in ticket.attachments],
            "missing_information": missing_information,
            "category": category,
        }
        if not missing_information:
            return {
                "resolved_information": {},
                "follow_up_questions": [],
                "recommended_actions": [],
                "notes": "No missing information.",
                "input_payload": input_payload,
            }
        if not self._api_key:
            self.last_secondary_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            return {
                "resolved_information": {},
                "follow_up_questions": [
                    f"OpenAI API key not configured. Please provide: {item}" for item in missing_information
                ],
                "recommended_actions": [],
                "notes": "Secondary enrichment skipped because OPENAI_API_KEY is missing.",
                "input_payload": input_payload,
            }

        prompt = self._build_secondary_prompt(ticket, missing_information, secondary_prompt)
        response = requests.post(
            "https://api.openai.com/v1/responses",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._model,
                "input": prompt,
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "ticket_secondary_enrichment",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "resolved_information": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "requirement": {"type": "string"},
                                            "value": {"type": "string"},
                                            "evidence": {"type": "string"},
                                        },
                                        "required": ["requirement", "value", "evidence"],
                                        "additionalProperties": False,
                                    },
                                },
                                "follow_up_questions": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "recommended_actions": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "notes": {"type": "string"},
                            },
                            "required": [
                                "resolved_information",
                                "follow_up_questions",
                                "recommended_actions",
                                "notes",
                            ],
                            "additionalProperties": False,
                        },
                    }
                },
            },
            timeout=45,
        )
        response.raise_for_status()
        payload = self._extract_json_object(response.json())
        self.last_secondary_usage = self._extract_usage(response.json())

        resolved_raw = payload.get("resolved_information", [])
        resolved_map: dict[str, dict[str, str]] = {}
        if isinstance(resolved_raw, list):
            for row in resolved_raw:
                if not isinstance(row, dict):
                    continue
                requirement = str(row.get("requirement", "")).strip()
                value = str(row.get("value", "")).strip()
                evidence = str(row.get("evidence", "")).strip()
                if requirement and value:
                    resolved_map[requirement] = {"value": value, "evidence": evidence}

        follow_up = payload.get("follow_up_questions", [])
        actions = payload.get("recommended_actions", [])
        notes = str(payload.get("notes", "")).strip()

        return {
            "resolved_information": resolved_map,
            "follow_up_questions": [str(item) for item in follow_up if str(item).strip()],
            "recommended_actions": [str(item) for item in actions if str(item).strip()],
            "notes": notes,
            "input_payload": input_payload,
            "token_usage": self.last_secondary_usage,
        }

    @staticmethod
    def _extract_usage(payload: dict[str, Any]) -> dict[str, int]:
        usage = payload.get("usage", {})
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

    @staticmethod
    def _extract_category(payload: dict[str, Any]) -> str:
        data = OpenAICategorizer._extract_json_object(payload)
        if isinstance(data, dict):
            return str(data.get("category", "other"))
        return "other"

    @staticmethod
    def _extract_json_object(payload: dict[str, Any]) -> dict[str, Any]:
        text_blob = payload.get("output_text")
        if isinstance(text_blob, str) and text_blob.strip():
            data = json.loads(text_blob)
            if isinstance(data, dict):
                return data

        output = payload.get("output", [])
        for item in output:
            for content in item.get("content", []):
                if content.get("type") == "output_text" and content.get("text"):
                    data = json.loads(content["text"])
                    if isinstance(data, dict):
                        return data
        return {}

    @staticmethod
    def _build_prompt(ticket: RawTicket, categories: list[str]) -> str:
        return (
            "Classify the Jira ticket into exactly one category. "
            f"Allowed categories: {', '.join(categories)}. "
            "Return JSON only.\n\n"
            f"Issue key: {ticket.issue_key}\n"
            f"Issue type: {ticket.issue_type}\n"
            f"Summary: {ticket.summary}\n"
            f"Description: {ticket.description}\n"
            f"Labels: {', '.join(ticket.labels)}\n"
        )

    @staticmethod
    def _build_secondary_prompt(
        ticket: RawTicket,
        missing_information: list[str],
        category_instructions: str,
    ) -> str:
        attachment_names = [att.filename for att in ticket.attachments]
        raw_ticket_json = json.dumps(ticket.raw, ensure_ascii=False)
        return (
            f"{category_instructions}\n\n"
            "Input payload below includes raw ticket information, attachment names, and missing information list.\n\n"
            f"raw_ticket: {raw_ticket_json}\n"
            f"attachment_names: {json.dumps(attachment_names, ensure_ascii=False)}\n"
            f"missing_information: {json.dumps(missing_information, ensure_ascii=False)}\n"
        )

    @staticmethod
    def _fallback_category(ticket: RawTicket) -> str:
        text = f"{ticket.issue_type} {ticket.summary} {ticket.description}".lower()
        if "incident" in text or "outage" in text or "sev" in text:
            return "incident"
        if "bug" in text or "error" in text or "failure" in text:
            return "bug"
        if "access" in text or "permission" in text:
            return "access_request"
        if "feature" in text or "enhancement" in text or "request" in text:
            return "feature_request"
        return "other"
