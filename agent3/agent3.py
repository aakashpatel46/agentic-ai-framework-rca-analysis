from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI
from pinecone import Pinecone


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


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def sparse_from_text(text: str, hash_buckets: int = 1 << 20) -> dict[str, list[int] | list[float]]:
    counts = Counter(tokenize(text))
    bucket_counts: dict[int, int] = {}
    for token, count in counts.items():
        idx = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % hash_buckets
        bucket_counts[idx] = bucket_counts.get(idx, 0) + count
    indices = sorted(bucket_counts.keys())
    values = [float(bucket_counts[i]) for i in indices]
    return {"indices": indices, "values": values}


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


@dataclass
class SimilarTicket:
    issue_key: str
    score: float
    summary: str
    ai_summary: str
    description: str
    organizations: str
    updated: str
    matched_vector_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "issue_key": self.issue_key,
            "score": round(self.score, 6),
            "summary": self.summary,
            "ai_summary": self.ai_summary,
            "description": self.description,
            "organizations": self.organizations,
            "updated": self.updated,
            "matched_vector_id": self.matched_vector_id,
        }


@dataclass
class DocumentationMatch:
    score: float
    path: str
    chunk_index: int
    chunk_count: int
    matched_vector_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 6),
            "path": self.path,
            "chunk_index": self.chunk_index,
            "chunk_count": self.chunk_count,
            "matched_vector_id": self.matched_vector_id,
        }


class Agent3Analyzer:
    def __init__(
        self,
        openai_client: OpenAI,
        pinecone_client: Pinecone,
        ticket_index_name: str,
        documentation_index_name: str,
        embedding_model: str = "text-embedding-3-small",
        embedding_dimensions: int | None = None,
        analysis_model: str = "gpt-4o-mini",
        namespace: str = "default",
        ticket_top_k: int = 15,
        documentation_top_k: int = 8,
    ) -> None:
        self.openai_client = openai_client
        self.pinecone_client = pinecone_client
        self.ticket_index_name = ticket_index_name
        self.documentation_index_name = documentation_index_name
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.analysis_model = analysis_model
        self.namespace = namespace
        self.ticket_top_k = ticket_top_k
        self.documentation_top_k = documentation_top_k
        self._token_calls: list[dict[str, Any]] = []

    @classmethod
    def from_env(cls) -> "Agent3Analyzer":
        embedding_dimensions_raw = (os.getenv("EMBEDDING_DIMENSIONS", "") or "").strip()
        embedding_dimensions = int(embedding_dimensions_raw) if embedding_dimensions_raw else None
        return cls(
            openai_client=OpenAI(api_key=get_env("OPENAI_API_KEY")),
            pinecone_client=Pinecone(api_key=get_env("PINECONE_API_KEY")),
            ticket_index_name=get_env("PINECONE_INDEX_NAME"),
            documentation_index_name=get_env("PINECONE_INDEX_DOCUMENTATION"),
            embedding_model=get_env("EMBEDDING_MODEL", required=False, default="text-embedding-3-small"),
            embedding_dimensions=embedding_dimensions,
            analysis_model=get_env("AGENT3_ANALYSIS_MODEL", required=False, default="gpt-4o-mini"),
            namespace=get_env("AGENT3_PINECONE_NAMESPACE", required=False, default="default"),
            ticket_top_k=int(get_env("AGENT3_TICKET_TOP_K", required=False, default="15")),
            documentation_top_k=int(get_env("AGENT3_DOC_TOP_K", required=False, default="8")),
        )

    def analyze_ticket(self, ticket: dict[str, Any]) -> dict[str, Any]:
        self._token_calls = []
        attachment_paths = self._extract_attachment_paths(ticket)
        attachment_analysis = self._analyze_attachments(attachment_paths)
        query_text = self._build_ticket_query_text(ticket, attachment_analysis)
        errors: list[str] = []
        try:
            ticket_matches_raw = self._query_index(self.ticket_index_name, query_text, self.ticket_top_k)
        except Exception as exc:
            ticket_matches_raw = []
            errors.append(f"ticket_index_query_failed: {exc}")

        try:
            doc_matches_raw = self._query_index(self.documentation_index_name, query_text, self.documentation_top_k)
        except Exception as exc:
            doc_matches_raw = []
            errors.append(f"documentation_index_query_failed: {exc}")

        current_issue_key = self._extract_ticket_key(ticket)
        similar_tickets = self._extract_top_similar_tickets(ticket_matches_raw, current_issue_key, top_n=3)
        documentation_matches = self._extract_doc_matches(doc_matches_raw, top_n=5)
        detailed_analysis = self._build_detailed_analysis(ticket, similar_tickets, documentation_matches, attachment_analysis)

        return {
            "agent": "agent3",
            "generated_at_utc": utc_now_iso(),
            "input_ticket": ticket,
            "query_context": {
                "ticket_index_name": self.ticket_index_name,
                "documentation_index_name": self.documentation_index_name,
                "namespace": self.namespace,
                "embedding_model": self.embedding_model,
            },
            "similar_tickets": [x.to_dict() for x in similar_tickets],
            "similar_tickets_top3": [x.to_dict() for x in similar_tickets],
            "documentation": [x.to_dict() for x in documentation_matches],
            "documentation_related": [x.to_dict() for x in documentation_matches],
            "logs": attachment_analysis.get("logs", []),
            "yml_files": attachment_analysis.get("yml_files", []),
            "other_attachments": attachment_analysis.get("other_attachments", []),
            "skipped_videos": attachment_analysis.get("skipped_videos", []),
            "attachment_summary": attachment_analysis.get("summary", {}),
            "detailed_analysis": detailed_analysis,
            "analysis_notes": [
                "similar_tickets_top3 is deduplicated by issue key from chunked vectors",
                "documentation index metadata currently returns file path/chunk pointers for follow-up review",
                "attachments are analyzed for non-video files and included in logs/yml_files sections",
            ],
            "token_usage": self._build_token_usage_summary(),
            "errors": errors,
        }

    def _query_index(self, index_name: str, query_text: str, top_k: int) -> list[dict[str, Any]]:
        dense = self._embed_text(query_text)
        sparse = sparse_from_text(query_text)
        index = self.pinecone_client.Index(index_name)
        try:
            response = index.query(
                vector=dense,
                sparse_vector=sparse,
                top_k=top_k,
                namespace=self.namespace,
                include_metadata=True,
            )
        except Exception as exc:
            # Some Pinecone indexes are dense-only and reject sparse vectors.
            if "sparse" not in str(exc).lower():
                raise
            response = index.query(
                vector=dense,
                top_k=top_k,
                namespace=self.namespace,
                include_metadata=True,
            )
        raw_matches = getattr(response, "matches", None)
        if raw_matches is None and isinstance(response, dict):
            raw_matches = response.get("matches", [])
        if not raw_matches:
            return []

        out: list[dict[str, Any]] = []
        for match in raw_matches:
            if isinstance(match, dict):
                out.append(match)
                continue
            out.append(
                {
                    "id": getattr(match, "id", ""),
                    "score": float(getattr(match, "score", 0.0)),
                    "metadata": getattr(match, "metadata", {}) or {},
                }
            )
        return out

    def _embed_text(self, text: str) -> list[float]:
        kwargs: dict[str, Any] = {
            "model": self.embedding_model,
            "input": [normalize_whitespace(text)],
        }
        if self.embedding_dimensions is not None:
            kwargs["dimensions"] = self.embedding_dimensions
        resp = self.openai_client.embeddings.create(**kwargs)
        self._record_usage("embedding", getattr(resp, "usage", None))
        return resp.data[0].embedding

    def _extract_ticket_key(self, ticket: dict[str, Any]) -> str:
        for key in ["Issue key", "issue_key", "ticket_key", "key", "id"]:
            value = str(ticket.get(key, "")).strip()
            if value:
                return value
        return ""

    def _extract_top_similar_tickets(
        self,
        matches: list[dict[str, Any]],
        current_issue_key: str,
        top_n: int,
    ) -> list[SimilarTicket]:
        dedup: dict[str, SimilarTicket] = {}
        for match in matches:
            metadata = match.get("metadata", {}) or {}
            vector_id = str(match.get("id", "")).strip()
            issue_key = (
                str(metadata.get("parent_issue_key", "")).strip()
                or str(metadata.get("issue_key", "")).strip()
                or vector_id.split("::")[0]
            )
            if not issue_key:
                continue
            if current_issue_key and issue_key == current_issue_key:
                continue

            score = float(match.get("score", 0.0))
            candidate = SimilarTicket(
                issue_key=issue_key,
                score=score,
                summary=normalize_whitespace(str(metadata.get("summary", ""))),
                ai_summary=normalize_whitespace(str(metadata.get("ai_summary", ""))),
                description=normalize_whitespace(str(metadata.get("description", ""))),
                organizations=normalize_whitespace(str(metadata.get("organizations", ""))),
                updated=normalize_whitespace(str(metadata.get("updated", ""))),
                matched_vector_id=vector_id,
            )
            existing = dedup.get(issue_key)
            if existing is None or candidate.score > existing.score:
                dedup[issue_key] = candidate

        top = sorted(dedup.values(), key=lambda x: x.score, reverse=True)
        return top[:top_n]

    def _extract_doc_matches(self, matches: list[dict[str, Any]], top_n: int) -> list[DocumentationMatch]:
        out: list[DocumentationMatch] = []
        for match in matches[:top_n]:
            metadata = match.get("metadata", {}) or {}
            out.append(
                DocumentationMatch(
                    score=float(match.get("score", 0.0)),
                    path=normalize_whitespace(str(metadata.get("path", ""))),
                    chunk_index=int(metadata.get("chunk_index", 0) or 0),
                    chunk_count=int(metadata.get("chunk_count", 0) or 0),
                    matched_vector_id=str(match.get("id", "")).strip(),
                )
            )
        return out

    def _build_ticket_query_text(self, ticket: dict[str, Any], attachment_analysis: dict[str, Any] | None = None) -> str:
        key_order = [
            "Issue key",
            "issue_key",
            "Summary",
            "summary",
            "Description",
            "description",
            "AI Summary",
            "ai_summary",
            "Priority",
            "priority",
            "Issue Type",
            "issue_type",
            "Components",
            "components",
            "Labels",
            "labels",
            "Organizations",
            "organizations",
            "Comments",
            "comments",
            "Steps to Reproduce",
            "steps_to_reproduce",
            "Environment",
            "environment",
            "Affects Version/s",
            "affects_version",
            "Error Message",
            "error_message",
            "Stack Trace",
            "stack_trace",
        ]
        lines: list[str] = []
        seen: set[str] = set()
        for key in key_order:
            if key in seen:
                continue
            seen.add(key)
            if key not in ticket:
                continue
            value = ticket.get(key)
            if value is None:
                continue
            if isinstance(value, list):
                cleaned = ", ".join([normalize_whitespace(str(v)) for v in value if str(v).strip()])
            elif isinstance(value, dict):
                cleaned = normalize_whitespace(json.dumps(value, ensure_ascii=False))
            else:
                cleaned = normalize_whitespace(str(value))
            if cleaned:
                lines.append(f"{key}: {cleaned}")

        if not lines:
            lines.append(normalize_whitespace(json.dumps(ticket, ensure_ascii=False)))
        if isinstance(attachment_analysis, dict):
            summary = attachment_analysis.get("summary", {})
            if isinstance(summary, dict):
                lines.append(
                    "Attachment Analysis Summary: "
                    f"total={summary.get('total', 0)}, "
                    f"analyzed_non_video={summary.get('analyzed_non_video', 0)}, "
                    f"logs={summary.get('logs', 0)}, "
                    f"yml_files={summary.get('yml_files', 0)}, "
                    f"skipped_videos={summary.get('skipped_videos', 0)}"
                )
            for row in attachment_analysis.get("logs", [])[:3]:
                if isinstance(row, dict):
                    lines.append(f"Log Attachment: {row.get('path', '')} | highlights={row.get('highlights', [])}")
            for row in attachment_analysis.get("yml_files", [])[:3]:
                if isinstance(row, dict):
                    lines.append(f"YML Attachment: {row.get('path', '')} | keys={row.get('top_level_keys', [])}")
        return "\n".join(lines)

    def _build_detailed_analysis(
        self,
        ticket: dict[str, Any],
        similar_tickets: list[SimilarTicket],
        documentation_matches: list[DocumentationMatch],
        attachment_analysis: dict[str, Any],
    ) -> dict[str, Any]:
        fallback = self._build_fallback_analysis(ticket, similar_tickets, documentation_matches, attachment_analysis)
        prompt_payload = {
            "current_ticket": ticket,
            "similar_tickets_top3": [x.to_dict() for x in similar_tickets],
            "documentation_related": [x.to_dict() for x in documentation_matches],
            "attachment_analysis": attachment_analysis,
            "instructions": {
                "focus": [
                    "probable root cause hypotheses",
                    "what to verify first",
                    "what immediate mitigations are reasonable",
                    "what longer-term fixes or guardrails to add",
                ]
            },
        }
        try:
            response = self.openai_client.responses.create(
                model=self.analysis_model,
                temperature=0.2,
                max_output_tokens=1200,
                input=[
                    {
                        "role": "system",
                        "content": (
                            "You are Agent3 in an RCA pipeline. Return ONLY valid JSON with keys: "
                            "executive_summary (string), probable_root_causes (array of strings), "
                            "recommended_diagnostics (array of strings), confidence (string), "
                            "supporting_signals (array of strings). "
                            "Do NOT include fixes, mitigation plans, rollback steps, or implementation actions."
                        ),
                    },
                    {"role": "user", "content": json.dumps(prompt_payload, ensure_ascii=False)},
                ],
            )
            self._record_usage("analysis", getattr(response, "usage", None))
            text = getattr(response, "output_text", "") or ""
            parsed = safe_json_loads(text)
            if parsed:
                parsed["analysis_source"] = "openai"
                return parsed
        except Exception as exc:
            fallback["analysis_error"] = f"LLM synthesis fallback used: {exc}"

        return fallback

    def _usage_to_dict(self, usage: Any) -> dict[str, int]:
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

    def _record_usage(self, call_name: str, usage: Any) -> None:
        usage_dict = self._usage_to_dict(usage)
        self._token_calls.append({"call": call_name, **usage_dict})

    def _build_token_usage_summary(self) -> dict[str, Any]:
        input_total = sum(int(x.get("input_tokens", 0) or 0) for x in self._token_calls)
        output_total = sum(int(x.get("output_tokens", 0) or 0) for x in self._token_calls)
        total = sum(int(x.get("total_tokens", 0) or 0) for x in self._token_calls)
        return {
            "calls": self._token_calls,
            "session_totals": {
                "input_tokens": input_total,
                "output_tokens": output_total,
                "total_tokens": total,
            },
        }

    def _build_fallback_analysis(
        self,
        ticket: dict[str, Any],
        similar_tickets: list[SimilarTicket],
        documentation_matches: list[DocumentationMatch],
        attachment_analysis: dict[str, Any],
    ) -> dict[str, Any]:
        ticket_summary = normalize_whitespace(
            str(ticket.get("Summary", "") or ticket.get("summary", "") or "No summary provided")
        )
        top_similar = similar_tickets[:3]
        top_doc_paths = [d.path for d in documentation_matches[:3] if d.path]

        supporting_signals: list[str] = []
        for item in top_similar:
            signal = item.ai_summary or item.summary or item.description
            if signal:
                supporting_signals.append(f"{item.issue_key}: {signal[:240]}")
        for path in top_doc_paths:
            supporting_signals.append(f"Documentation pointer: {path}")
        for row in attachment_analysis.get("logs", [])[:2]:
            if isinstance(row, dict):
                supporting_signals.append(
                    f"Log signal ({row.get('path', '')}): {str(row.get('highlights', []))[:220]}"
                )
        for row in attachment_analysis.get("yml_files", [])[:2]:
            if isinstance(row, dict):
                supporting_signals.append(
                    f"YML signal ({row.get('path', '')}): keys={row.get('top_level_keys', [])}"
                )

        diagnostics = [
            "Reproduce the issue in a lower environment using ticket steps, then capture logs and stack trace.",
            "Compare behavior with top similar tickets and verify whether the same component/version is involved.",
            "Review recent code/config changes around components referenced in the current ticket.",
        ]
        if top_doc_paths:
            diagnostics.append("Validate runtime/config against matched documentation pages before patching code.")

        return {
            "analysis_source": "fallback",
            "executive_summary": (
                f"Initial RCA analysis for '{ticket_summary}'. The issue appears related to previously resolved patterns "
                f"and should be triaged against similar incidents and matched documentation."
            ),
            "probable_root_causes": [
                "Regression or configuration drift in a shared component touched by recent changes.",
                "Unhandled edge case similar to historical incidents identified from ticket index.",
                "Environment-specific mismatch (version/config/data state) causing inconsistent behavior.",
            ],
            "recommended_diagnostics": diagnostics,
            "confidence": "medium",
            "supporting_signals": supporting_signals,
        }

    @staticmethod
    def _extract_attachment_paths(ticket: dict[str, Any]) -> list[str]:
        candidates = []
        for key in ["Attachment Paths", "attachment_paths", "attachments", "attachmentPaths"]:
            value = ticket.get(key)
            if isinstance(value, list):
                candidates.extend([str(x).strip() for x in value if str(x).strip()])
        deduped: list[str] = []
        seen: set[str] = set()
        for path in candidates:
            norm = path.lower()
            if norm not in seen:
                seen.add(norm)
                deduped.append(path)
        return deduped

    @staticmethod
    def _is_video_file(path: str) -> bool:
        ext = Path(path).suffix.lower()
        return ext in {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".webm", ".m4v", ".3gp", ".mpeg", ".mpg"}

    @staticmethod
    def _is_yml_file(path: str) -> bool:
        ext = Path(path).suffix.lower()
        return ext in {".yml", ".yaml"}

    @staticmethod
    def _is_log_file(path: str) -> bool:
        ext = Path(path).suffix.lower()
        name = Path(path).name.lower()
        return ext == ".log" or "log" in name

    @staticmethod
    def _is_text_file(path: str) -> bool:
        ext = Path(path).suffix.lower()
        return ext in {
            ".txt",
            ".log",
            ".yml",
            ".yaml",
            ".json",
            ".xml",
            ".csv",
            ".md",
            ".ini",
            ".cfg",
            ".conf",
            ".properties",
            ".sql",
            ".sh",
            ".ps1",
            ".bat",
            ".java",
            ".py",
            ".js",
            ".ts",
        }

    @staticmethod
    def _read_text_excerpt(path: Path, max_chars: int = 4000) -> str:
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            return normalize_whitespace(content[:max_chars])
        except Exception:
            return ""

    def _analyze_attachments(self, attachment_paths: list[str]) -> dict[str, Any]:
        logs: list[dict[str, Any]] = []
        yml_files: list[dict[str, Any]] = []
        other_attachments: list[dict[str, Any]] = []
        skipped_videos: list[str] = []
        analyzed_non_video = 0

        for raw_path in attachment_paths:
            p = Path(raw_path)
            if self._is_video_file(raw_path):
                skipped_videos.append(str(p))
                continue

            analyzed_non_video += 1
            exists = p.exists()
            row_base = {
                "path": str(p),
                "exists": exists,
                "size_bytes": int(p.stat().st_size) if exists else 0,
            }
            if self._is_log_file(raw_path):
                excerpt = self._read_text_excerpt(p) if exists else ""
                highlights = []
                if excerpt:
                    for kw in ["error", "exception", "failed", "timeout", "nullpointer", "stacktrace"]:
                        if kw in excerpt.lower():
                            highlights.append(kw)
                logs.append(
                    {
                        **row_base,
                        "highlights": highlights,
                        "excerpt": excerpt[:1200],
                    }
                )
                continue

            if self._is_yml_file(raw_path):
                excerpt = self._read_text_excerpt(p) if exists else ""
                top_level_keys: list[str] = []
                if excerpt:
                    for line in excerpt.splitlines():
                        m = re.match(r"^\s*([A-Za-z0-9_.-]+)\s*:\s*(?:#.*)?$", line)
                        if m:
                            k = m.group(1)
                            if k not in top_level_keys:
                                top_level_keys.append(k)
                        if len(top_level_keys) >= 20:
                            break
                yml_files.append(
                    {
                        **row_base,
                        "top_level_keys": top_level_keys,
                        "excerpt": excerpt[:1200],
                    }
                )
                continue

            if self._is_text_file(raw_path):
                excerpt = self._read_text_excerpt(p) if exists else ""
                other_attachments.append(
                    {
                        **row_base,
                        "type": "text",
                        "excerpt": excerpt[:600],
                    }
                )
            else:
                other_attachments.append(
                    {
                        **row_base,
                        "type": "binary_or_non_text",
                        "note": "Attachment considered for analysis metadata only.",
                    }
                )

        return {
            "summary": {
                "total": len(attachment_paths),
                "analyzed_non_video": analyzed_non_video,
                "logs": len(logs),
                "yml_files": len(yml_files),
                "other_attachments": len(other_attachments),
                "skipped_videos": len(skipped_videos),
            },
            "logs": logs,
            "yml_files": yml_files,
            "other_attachments": other_attachments,
            "skipped_videos": skipped_videos,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agent3: Similar ticket + documentation analysis")
    parser.add_argument(
        "--ticket-json-file",
        default=os.getenv("AGENT3_INPUT_TICKET_JSON_FILE", ""),
        help="Path to current ticket JSON input file",
    )
    parser.add_argument(
        "--ticket-json-inline",
        default=os.getenv("AGENT3_INPUT_TICKET_JSON", ""),
        help="Raw JSON string for current ticket input",
    )
    parser.add_argument(
        "--output-file",
        default=os.getenv("AGENT3_OUTPUT_FILE", ""),
        help="Output JSON file path",
    )
    return parser.parse_args()


def load_ticket_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.ticket_json_file:
        path = Path(args.ticket_json_file)
        if not path.exists():
            raise FileNotFoundError(f"Ticket JSON file not found: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Ticket JSON file must contain a JSON object")
        return value

    if args.ticket_json_inline:
        value = json.loads(args.ticket_json_inline)
        if not isinstance(value, dict):
            raise ValueError("Inline ticket JSON must be a JSON object")
        return value

    raise ValueError("Provide --ticket-json-file or --ticket-json-inline")


def resolve_output_path(ticket: dict[str, Any], output_file: str) -> Path:
    if output_file.strip():
        return Path(output_file.strip())
    issue_key = (
        str(ticket.get("Issue key", "")).strip()
        or str(ticket.get("issue_key", "")).strip()
        or "ticket"
    )
    safe_issue_key = re.sub(r"[^a-zA-Z0-9_.-]", "_", issue_key)
    return Path(f"agent3/output/{safe_issue_key}.json")


def main() -> None:
    load_env_file(".env")
    args = parse_args()
    ticket = load_ticket_from_args(args)
    analyzer = Agent3Analyzer.from_env()
    result = analyzer.analyze_ticket(ticket)

    output_path = resolve_output_path(ticket, args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[agent3] Analysis complete")
    print(f"[agent3] Output JSON: {output_path.resolve()}")
    print(f"[agent3] Similar tickets found: {len(result['similar_tickets_top3'])}")
    print(f"[agent3] Documentation matches found: {len(result['documentation_related'])}")


if __name__ == "__main__":
    main()
