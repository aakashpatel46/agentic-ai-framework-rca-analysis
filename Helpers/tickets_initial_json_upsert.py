# for first run 
    # python Helpers\tickets_initial_json_upsert.py --tickets-json Helpers/files/JMCH_tickets_raw_json.json



from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from openai import BadRequestError
from pinecone import Pinecone

DEFAULT_TICKETS_JSON = "Helpers/files/JMCH_tickets_raw_json.json"
DEFAULT_STATE_FILE = "Helpers/files/pinecone_tickets_state.json"
OPENAI_EMBED_BATCH_SIZE = 64
PINECONE_UPSERT_BATCH_SIZE = 100
SPARSE_HASH_BUCKETS = 1 << 20
MAX_SECTION_TOKENS = 1000
MAX_TOKENS_PER_CHUNK = 6000
OPENAI_EMBED_TOKEN_HARD_LIMIT = 8192
OPENAI_EMBED_SAFE_TOKENS = 7000


def get_env(name: str, required: bool = True, default: str | None = None) -> str:
    value = (os.getenv(name, default) or "").strip().strip('"').strip("'")
    if required and not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def sparse_from_text(text: str) -> dict[str, list[int] | list[float]]:
    counts = Counter(tokenize(text))
    bucket_counts: dict[int, int] = {}
    for token, count in counts.items():
        idx = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % SPARSE_HASH_BUCKETS
        bucket_counts[idx] = bucket_counts.get(idx, 0) + count
    indices = sorted(bucket_counts.keys())
    values = [float(bucket_counts[i]) for i in indices]
    return {"indices": indices, "values": values}


def estimate_tokens(text: str) -> int:
    # Conservative heuristic to avoid hitting embedding hard limits.
    # Tokens are usually ~3-4 chars in English prose; use 3.2 for safety.
    return max(1, int(len(text) / 3.2))


def split_text_by_token_budget(text: str, max_tokens: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= max_tokens:
        return [" ".join(words)]
    chunks: list[str] = []
    for i in range(0, len(words), max_tokens):
        chunks.append(" ".join(words[i : i + max_tokens]))
    return chunks


def sanitize_embedding_text(text: str) -> str:
    # Remove problematic characters that can break JSON serialization downstream.
    text = text.replace("\x00", " ")
    text = re.sub(r"[\ud800-\udfff]", " ", text)
    text = normalize_whitespace(text)
    return text


def average_vectors(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    acc = [0.0] * dim
    for v in vectors:
        for i in range(dim):
            acc[i] += v[i]
    n = float(len(vectors))
    return [x / n for x in acc]


def embed_text_resilient(
    openai_client: OpenAI,
    model: str,
    text: str,
    dimensions: int | None = None,
) -> list[float]:
    text = sanitize_embedding_text(text)
    kwargs: dict[str, Any] = {"model": model, "input": [text]}
    if dimensions is not None:
        kwargs["dimensions"] = dimensions
    try:
        return openai_client.embeddings.create(**kwargs).data[0].embedding
    except BadRequestError as e:
        msg = str(e).lower()
        if "parse the json body" in msg:
            cleaned = sanitize_embedding_text(text)
            if cleaned != text:
                kwargs["input"] = [cleaned]
                return openai_client.embeddings.create(**kwargs).data[0].embedding
            raise
        if "maximum context length" not in msg and "maximum input length" not in msg and "8192" not in msg:
            raise

        words = text.split()
        if len(words) <= 50:
            raise
        half = len(words) // 2
        left = " ".join(words[:half]).strip()
        right = " ".join(words[half:]).strip()
        parts = [p for p in [left, right] if p]
        if len(parts) < 2:
            raise
        subvectors = [embed_text_resilient(openai_client, model, p, dimensions=dimensions) for p in parts]
        return average_vectors(subvectors)


def enforce_embedding_limit(records: list[dict[str, Any]], safe_tokens: int = OPENAI_EMBED_SAFE_TOKENS) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for record in records:
        est = estimate_tokens(record["text"])
        if est <= safe_tokens:
            out.append(record)
            continue

        # Split oversized chunk into subchunks and preserve retrievable metadata.
        subtexts = split_text_by_token_budget(record["text"], safe_tokens)
        if len(subtexts) <= 1:
            out.append(record)
            continue

        parent_id = record["id"]
        parent_md = dict(record["metadata"])
        for i, subtext in enumerate(subtexts, start=1):
            md = dict(parent_md)
            md["subchunk_index"] = i
            md["subchunk_total"] = len(subtexts)
            out.append(
                {
                    "id": f"{parent_id}::part-{i}",
                    "text": subtext,
                    "metadata": md,
                }
            )
    return out


def ticket_to_records(ticket: dict[str, Any]) -> list[dict[str, Any]]:
    issue_key = str(ticket.get("Issue key", "")).strip()
    if not issue_key:
        return []

    summary = normalize_whitespace(str(ticket.get("Summary", "")))
    description = normalize_whitespace(str(ticket.get("Description", "")))
    org_raw = ticket.get("Organizations", [])
    organizations = ", ".join([str(o) for o in org_raw]) if isinstance(org_raw, list) else str(org_raw)
    organizations = normalize_whitespace(organizations)
    ai_summary = normalize_whitespace(str(ticket.get("AI Summary", ticket.get("ai_summary", ""))))
    comments_raw = ticket.get("Comments", [])
    comments = [normalize_whitespace(str(c)) for c in comments_raw] if isinstance(comments_raw, list) else []

    section_units: list[dict[str, Any]] = []

    def add_section(kind: str, label: str, text: str, comment_index: int = 0) -> None:
        if not text:
            return
        parts = split_text_by_token_budget(text, MAX_SECTION_TOKENS)
        for part_idx, part in enumerate(parts, start=1):
            display = label if len(parts) == 1 else f"{label} (part {part_idx})"
            section_units.append(
                {
                    "kind": kind,
                    "label": display,
                    "text": part,
                    "token_count": estimate_tokens(part),
                    "comment_index": comment_index,
                }
            )

    add_section("summary", "Summary", summary)
    add_section("description", "Description", description)
    add_section("organizations", "Organizations", organizations)
    add_section("ai_summary", "AI Summary", ai_summary)
    for i, comment in enumerate(comments, start=1):
        add_section("comment", f"Comment {i}", comment, comment_index=i)

    if not section_units:
        return []

    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_tokens = 0

    for unit in section_units:
        unit_tokens = unit["token_count"]
        if current and current_tokens + unit_tokens > MAX_TOKENS_PER_CHUNK:
            chunks.append(current)
            current = []
            current_tokens = 0
        current.append(unit)
        current_tokens += unit_tokens
    if current:
        chunks.append(current)

    records: list[dict[str, Any]] = []
    chunk_total = len(chunks)
    for idx, chunk_units in enumerate(chunks, start=1):
        chunk_text = "\n\n".join([f"{u['label']}:\n{u['text']}" for u in chunk_units]).strip()
        vector_id = issue_key if idx == 1 else f"{issue_key}::chunk-{idx}"

        chunk_comments = [f"{u['label']}: {u['text']}" for u in chunk_units if u["kind"] == "comment"]
        records.append(
            {
                "id": vector_id,
                "text": chunk_text,
                "metadata": {
                    "type": "ticket",
                    "source": "local_json_initial",
                    "issue_key": issue_key,
                    "parent_issue_key": issue_key,
                    "chunk_index": idx,
                    "chunk_total": chunk_total,
                    "summary": summary if idx == 1 else "",
                    "description": description if idx == 1 else "",
                    "organizations": organizations if idx == 1 else "",
                    "ai_summary": ai_summary if idx == 1 else "",
                    "comments": chunk_comments,
                    "updated": str(ticket.get("Updated", "")),
                },
            }
        )

    return records


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"last_jira_upsert_timestamp_utc": "", "last_run_completed_utc": "", "upserted_vector_ids": []}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    upserted_ids = data.get("upserted_vector_ids", [])
    if not isinstance(upserted_ids, list):
        upserted_ids = []
    return {
        "last_jira_upsert_timestamp_utc": str(data.get("last_jira_upsert_timestamp_utc", "")),
        "last_run_completed_utc": str(data.get("last_run_completed_utc", "")),
        "upserted_vector_ids": upserted_ids,
    }


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="First-time ticket upsert from local JSON into PINECONE_INDEX_NAME (hybrid vectors)."
    )
    p.add_argument("--tickets-json", default=DEFAULT_TICKETS_JSON)
    p.add_argument("--state-file", default=DEFAULT_STATE_FILE)
    p.add_argument("--namespace", default="default")
    p.add_argument(
        "--dont-set-last-upsert-now",
        action="store_true",
        help="Do not set last_jira_upsert_timestamp_utc after initial JSON upsert.",
    )
    return p.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    logging.basicConfig(
        level=os.getenv("PINECONE_UPSERT_LOG_LEVEL", "INFO"),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger("tickets_initial_json_upsert")

    openai_client = OpenAI(api_key=get_env("OPENAI_API_KEY"))
    embedding_model = get_env("EMBEDDING_MODEL", required=False, default="text-embedding-3-small")
    embedding_dimensions_raw = (os.getenv("EMBEDDING_DIMENSIONS", "") or "").strip()
    embedding_dimensions = int(embedding_dimensions_raw) if embedding_dimensions_raw else None

    pinecone_client = Pinecone(api_key=get_env("PINECONE_API_KEY"))
    index_name = get_env("PINECONE_INDEX_NAME")
    index = pinecone_client.Index(index_name)

    index_dimension = None
    try:
        desc = pinecone_client.describe_index(index_name)
        index_dimension = getattr(desc, "dimension", None)
    except Exception:
        pass

    if embedding_dimensions is None and isinstance(index_dimension, int) and embedding_model.startswith("text-embedding-3"):
        embedding_dimensions = index_dimension
        logger.info(
            "Auto-setting embedding dimensions to index dimension: %s",
            embedding_dimensions,
        )

    probe_vec = embed_text_resilient(
        openai_client,
        embedding_model,
        "dimension probe",
        dimensions=embedding_dimensions,
    )
    if isinstance(index_dimension, int) and len(probe_vec) != index_dimension:
        raise RuntimeError(
            f"Embedding dimension {len(probe_vec)} does not match Pinecone index dimension {index_dimension}. "
            "Set EMBEDDING_DIMENSIONS in .env to match your index (for text-embedding-3 models)."
        )

    tickets_path = Path(args.tickets_json)
    with tickets_path.open("r", encoding="utf-8") as f:
        tickets = json.load(f)
    if not isinstance(tickets, list):
        raise ValueError(f"{tickets_path} must be a JSON list of tickets.")

    records: list[dict[str, Any]] = []
    for t in tickets:
        if isinstance(t, dict):
            records.extend(ticket_to_records(t))

    dedup = {r["id"]: r for r in records}
    final_records = list(dedup.values())
    final_records = enforce_embedding_limit(final_records)
    logger.info("Ticket vectors from JSON to upsert (after chunking+safety split): %s", len(final_records))
    if not final_records:
        logger.info("No tickets to upsert from JSON.")
        return

    state_path = Path(args.state_file)
    state = load_state(state_path)
    upserted_ids = set([str(x) for x in state.get("upserted_vector_ids", []) if str(x).strip()])
    pending_records = [r for r in final_records if r["id"] not in upserted_ids]
    logger.info("Vectors already upserted (skipped): %s", len(final_records) - len(pending_records))
    logger.info("Vectors pending upsert: %s", len(pending_records))

    total = len(pending_records)
    processed = 0
    for i in range(0, len(pending_records), PINECONE_UPSERT_BATCH_SIZE):
        record_batch = pending_records[i : i + PINECONE_UPSERT_BATCH_SIZE]
        payload_batch = []
        for rec in record_batch:
            clean_text = sanitize_embedding_text(rec["text"])
            dense = embed_text_resilient(
                openai_client,
                embedding_model,
                clean_text,
                dimensions=embedding_dimensions,
            )
            payload_batch.append(
                {
                    "id": rec["id"],
                    "values": dense,
                    "sparse_values": sparse_from_text(clean_text),
                    "metadata": rec["metadata"],
                }
            )

        index.upsert(vectors=payload_batch, namespace=args.namespace)
        for rec in record_batch:
            upserted_ids.add(rec["id"])

        processed += len(record_batch)
        state["upserted_vector_ids"] = sorted(upserted_ids)
        state["last_run_completed_utc"] = utc_now_iso()
        save_state(state_path, state)
        logger.info("Upserted %s/%s pending vectors", processed, total)

    if not args.dont_set_last_upsert_now:
        state["last_jira_upsert_timestamp_utc"] = utc_now_iso()
    state["last_run_completed_utc"] = utc_now_iso()
    state["upserted_vector_ids"] = sorted(upserted_ids)
    save_state(state_path, state)
    logger.info("Initial JSON upsert complete. State updated at %s", state_path)


if __name__ == "__main__":
    main()
