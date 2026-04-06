from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import re

from agent2.category_prompt_config import CategoryPromptConfig
from agent2.models import RawTicket, ValidationEnrichmentResult
from agent2.openai_categorizer import OpenAICategorizer
from agent2.rules import RuleSet
from agent2.source import TicketSource


class ValidationEnrichmentProcessor:
    def __init__(
        self,
        source: TicketSource,
        rules: RuleSet,
        categorizer: OpenAICategorizer,
        prompt_config: CategoryPromptConfig,
    ) -> None:
        self._source = source
        self._rules = rules
        self._categorizer = categorizer
        self._prompt_config = prompt_config

    def process_issue(self, issue_key: str, attachment_root: str) -> ValidationEnrichmentResult:
        ticket = self._source.fetch_issue(issue_key)
        attachment_paths = self._source.download_attachments(ticket, attachment_root)

        category = self._categorizer.categorize(ticket)
        required = self._rules.requirements_for(category)
        available = self._build_available_information(ticket, attachment_paths)
        missing = [item for item in required if not self._is_requirement_satisfied(item, ticket, attachment_paths, available)]
        secondary_called = False
        secondary_enrichment: dict[str, object] = {}
        if missing:
            secondary_called = True
            secondary_prompt = self._prompt_config.secondary_prompt_for(category)
            secondary_enrichment = self._categorizer.enrich_missing_information(
                ticket=ticket,
                category=category,
                missing_information=missing,
                secondary_prompt=secondary_prompt,
            )

        raw_ticket = asdict(ticket)
        return ValidationEnrichmentResult(
            source=self._source.name,
            issue_key=ticket.issue_key,
            category=category,
            required_information=required,
            available_information=sorted(available),
            missing_information=missing,
            attachment_paths=attachment_paths,
            secondary_openai_called=secondary_called,
            secondary_enrichment=secondary_enrichment,
            raw_ticket=raw_ticket,
        )

    @staticmethod
    def _build_available_information(ticket: RawTicket, attachment_paths: list[str]) -> set[str]:
        available: set[str] = set()

        field_map = {
            "summary": ticket.summary,
            "description": ticket.description,
            "steps_to_reproduce": ticket.steps_to_reproduce,
            "version": ticket.affects_version,
            "issue_type": ticket.issue_type,
            "status": ticket.status,
            "priority": ticket.priority,
            "reporter": ticket.reporter,
            "assignee": ticket.assignee,
        }
        for field_name, value in field_map.items():
            if value:
                available.add(field_name)

        if ValidationEnrichmentProcessor._has_steps_to_reproduce(ticket):
            available.add("reproduction steps")
            available.add("steps to reproduce")

        if ticket.labels:
            available.add("labels")
        if ticket.organizations:
            available.add("organizations")
        if ticket.comments:
            available.add("comments")

        has_logs = False
        for path in attachment_paths:
            file_name = path.split("/")[-1].split("\\")[-1].lower()
            available.add("attachment")
            available.add(f"attachment_file:{file_name}")
            if file_name.endswith(".log") or "log" in file_name:
                has_logs = True

        if has_logs:
            available.add("logs")
            available.add("log file")
            available.add("log")

        # Semantic availability from ticket text and readable attachment content.
        text_corpus = " ".join(
            [
                ticket.summary or "",
                ticket.description or "",
                ticket.steps_to_reproduce or "",
                ticket.affects_version or "",
                " ".join(ticket.labels or []),
                " ".join(ticket.organizations or []),
                " ".join(ticket.comments or []),
            ]
        )
        text_corpus += " " + ValidationEnrichmentProcessor._read_attachment_text_for_signals(attachment_paths)
        available.update(ValidationEnrichmentProcessor._extract_signal_tokens(text_corpus))

        return available

    @staticmethod
    def _is_requirement_satisfied(
        requirement: str,
        ticket: RawTicket,
        attachment_paths: list[str],
        available_information: set[str],
    ) -> bool:
        req = requirement.strip().lower()
        req_norm = ValidationEnrichmentProcessor._normalize_requirement(req)
        if req.startswith("attachment:"):
            expected = req.split(":", 1)[1]
            return any(expected in path.lower() for path in attachment_paths)

        if req in {"reproduction steps", "steps to reproduce", "repro steps"}:
            return ValidationEnrichmentProcessor._has_steps_to_reproduce(ticket)

        if req in {"logs", "log", "log file", "log files"}:
            return any(
                path.lower().endswith(".log")
                or "log" in path.lower().split("/")[-1].split("\\")[-1]
                for path in attachment_paths
            )

        if req in {"version", "versions", "affects version", "affects versions"}:
            return bool(ticket.affects_version and ticket.affects_version.strip())

        if any(req in path.lower() for path in attachment_paths):
            return True

        return req in available_information or req_norm in available_information

    @staticmethod
    def _has_steps_to_reproduce(ticket: RawTicket) -> bool:
        if ticket.steps_to_reproduce and ticket.steps_to_reproduce.strip():
            return True

        description = ticket.description
        if not description:
            return False

        match = re.search(
            r"(?:steps?\s+to\s+reproduce|reproduction\s+steps)\s*:?\s*(.*)",
            description,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not match:
            return False

        after_header = match.group(1).strip()
        if not after_header:
            return False

        stop_markers = (
            "actual result",
            "expected result",
            "environment",
            "product",
            "root cause",
        )
        lowered = after_header.lower()
        cut_index = len(after_header)
        for marker in stop_markers:
            idx = lowered.find(marker)
            if idx != -1:
                cut_index = min(cut_index, idx)
        steps_block = after_header[:cut_index].strip()
        if not steps_block:
            return False

        non_empty_lines = [line.strip(" -*\t") for line in steps_block.splitlines() if line.strip()]
        return any(len(line) >= 3 for line in non_empty_lines)

    @staticmethod
    def _normalize_requirement(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")

    @staticmethod
    def _extract_signal_tokens(text: str) -> set[str]:
        lowered = (text or "").lower()
        tokens = re.findall(r"[a-z0-9_./-]+", lowered)
        out = set(tokens)
        for token in list(tokens):
            out.add(ValidationEnrichmentProcessor._normalize_requirement(token))
        return {x for x in out if x}

    @staticmethod
    def _is_probably_text_file(path: Path) -> bool:
        return path.suffix.lower() in {
            ".txt",
            ".log",
            ".md",
            ".json",
            ".xml",
            ".yml",
            ".yaml",
            ".csv",
            ".ini",
            ".cfg",
            ".conf",
            ".properties",
            ".sql",
            ".py",
            ".java",
            ".js",
            ".ts",
            ".sh",
            ".ps1",
            ".bat",
        }

    @staticmethod
    def _read_attachment_text_for_signals(attachment_paths: list[str], max_chars_per_file: int = 4000) -> str:
        chunks: list[str] = []
        for raw_path in attachment_paths:
            path = Path(raw_path)
            if not path.exists() or not ValidationEnrichmentProcessor._is_probably_text_file(path):
                continue
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if content:
                chunks.append(content[:max_chars_per_file])
        return "\n".join(chunks)
