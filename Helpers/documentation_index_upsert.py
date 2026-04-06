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

DEFAULT_DOCS_ROOT = "Helpers/files/knowlede_source"
DEFAULT_STATE_FILE = "Helpers/files/pinecone_docs_state.json"
OPENAI_EMBED_BATCH_SIZE = 64
PINECONE_UPSERT_BATCH_SIZE = 100
DOC_CHUNK_WORDS = 350
DOC_CHUNK_OVERLAP_WORDS = 60
SPARSE_HASH_BUCKETS = 1 << 20
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


def strip_html(html: str) -> str:
    html = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
    html = re.sub(r"(?is)<[^>]+>", " ", html)
    return normalize_whitespace(html.replace("&nbsp;", " ").replace("&amp;", "&"))


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def sanitize_embedding_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[\ud800-\udfff]", " ", text)
    return normalize_whitespace(text)


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / 3.2))


def split_text_by_token_budget(text: str, max_tokens: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= max_tokens:
        return [" ".join(words)]
    return [" ".join(words[i : i + max_tokens]) for i in range(0, len(words), max_tokens)]


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
            kwargs["input"] = [cleaned]
            return openai_client.embeddings.create(**kwargs).data[0].embedding
        if "maximum context length" not in msg and "maximum input length" not in msg and "8192" not in msg:
            raise
        parts = split_text_by_token_budget(text, OPENAI_EMBED_SAFE_TOKENS)
        if len(parts) <= 1:
            raise
        subvectors = [embed_text_resilient(openai_client, model, p, dimensions=dimensions) for p in parts]
        return average_vectors(subvectors)


def sparse_from_text(text: str) -> dict[str, list[int] | list[float]]:
    counts = Counter(tokenize(text))
    bucket_counts: dict[int, int] = {}
    for token, count in counts.items():
        idx = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % SPARSE_HASH_BUCKETS
        bucket_counts[idx] = bucket_counts.get(idx, 0) + count
    indices = sorted(bucket_counts.keys())
    values = [float(bucket_counts[i]) for i in indices]
    return {"indices": indices, "values": values}


def chunk_text(text: str, chunk_words: int = DOC_CHUNK_WORDS, overlap_words: int = DOC_CHUNK_OVERLAP_WORDS) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [" ".join(words)]
    step = max(1, chunk_words - overlap_words)
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += step
    return chunks


def eligible_doc_file(path: Path) -> bool:
    return path.suffix.lower() in {".txt", ".md", ".html", ".htm", ".json", ".xml", ".yaml", ".yml"}


def file_fingerprint(path: Path) -> dict[str, Any]:
    st = path.stat()
    return {"mtime": int(st.st_mtime), "size": int(st.st_size)}


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"last_run_completed_utc": "", "docs_manifest": {}, "upserted_vector_ids": []}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    manifest = data.get("docs_manifest", {})
    if not isinstance(manifest, dict):
        manifest = {}
    upserted_ids = data.get("upserted_vector_ids", [])
    if not isinstance(upserted_ids, list):
        upserted_ids = []
    return {
        "last_run_completed_utc": str(data.get("last_run_completed_utc", "")),
        "docs_manifest": manifest,
        "upserted_vector_ids": upserted_ids,
    }


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upsert documentation to PINECONE_INDEX_DOCUMENTATION with hybrid vectors.")
    p.add_argument("--docs-root", default=DEFAULT_DOCS_ROOT)
    p.add_argument("--state-file", default=DEFAULT_STATE_FILE)
    p.add_argument("--namespace", default="default")
    p.add_argument("--force-full", action="store_true")
    return p.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    logging.basicConfig(level=os.getenv("PINECONE_UPSERT_LOG_LEVEL", "INFO"), format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("documentation_index_upsert")

    openai_client = OpenAI(api_key=get_env("OPENAI_API_KEY"))
    embedding_model = get_env("EMBEDDING_MODEL", required=False, default="text-embedding-3-small")
    embedding_dimensions_raw = (os.getenv("EMBEDDING_DIMENSIONS", "") or "").strip()
    embedding_dimensions = int(embedding_dimensions_raw) if embedding_dimensions_raw else None
    pinecone_client = Pinecone(api_key=get_env("PINECONE_API_KEY"))
    index_name = get_env("PINECONE_INDEX_DOCUMENTATION")
    index = pinecone_client.Index(index_name)

    index_dimension = None
    try:
        desc = pinecone_client.describe_index(index_name)
        index_dimension = getattr(desc, "dimension", None)
    except Exception:
        pass

    if embedding_dimensions is None and isinstance(index_dimension, int) and embedding_model.startswith("text-embedding-3"):
        embedding_dimensions = index_dimension
        logger.info("Auto-setting embedding dimensions to index dimension: %s", embedding_dimensions)

    probe = embed_text_resilient(openai_client, embedding_model, "dimension probe", dimensions=embedding_dimensions)
    if isinstance(index_dimension, int) and len(probe) != index_dimension:
        raise RuntimeError(
            f"Embedding dimension {len(probe)} does not match Pinecone index dimension {index_dimension}. "
            "Set EMBEDDING_DIMENSIONS in .env to match the docs index."
        )

    state_path = Path(args.state_file)
    state = load_state(state_path)
    old_manifest: dict[str, Any] = state.get("docs_manifest", {})
    next_manifest: dict[str, Any] = {}

    docs_root = Path(args.docs_root)
    if not docs_root.exists():
        raise FileNotFoundError(f"Documentation root not found: {docs_root}")

    records: list[dict[str, Any]] = []
    files = [p for p in docs_root.rglob("*") if p.is_file() and eligible_doc_file(p)]
    for file_path in files:
        rel = file_path.as_posix()
        fp = file_fingerprint(file_path)
        next_manifest[rel] = fp
        if not args.force_full and old_manifest.get(rel) == fp:
            continue

        raw = file_path.read_text(encoding="utf-8", errors="ignore")
        text = strip_html(raw) if file_path.suffix.lower() in {".html", ".htm"} else normalize_whitespace(raw)
        if not text:
            continue
        chunks = chunk_text(text)
        for idx, chunk in enumerate(chunks):
            key = f"{rel}::chunk::{idx}"
            records.append(
                {
                    "id": "doc:" + hashlib.sha1(key.encode("utf-8")).hexdigest(),
                    "text": chunk,
                    "metadata": {
                        "type": "doc",
                        "source": "documentation",
                        "path": rel,
                        "chunk_index": idx,
                        "chunk_count": len(chunks),
                        "file_mtime": fp["mtime"],
                    },
                }
            )

    logger.info("Documentation chunks to upsert: %s", len(records))
    if not records:
        state["docs_manifest"] = next_manifest
        state["last_run_completed_utc"] = utc_now_iso()
        save_state(state_path, state)
        logger.info("No changed docs. State refreshed: %s", state_path)
        return

    upserted_ids = set([str(x) for x in state.get("upserted_vector_ids", []) if str(x).strip()])
    pending_records = [r for r in records if r["id"] not in upserted_ids]
    logger.info("Documentation vectors already upserted (skipped): %s", len(records) - len(pending_records))
    logger.info("Documentation vectors pending upsert: %s", len(pending_records))

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
        logger.info("Upserted %s/%s pending documentation vectors", processed, total)

    state["docs_manifest"] = next_manifest
    state["last_run_completed_utc"] = utc_now_iso()
    state["upserted_vector_ids"] = sorted(upserted_ids)
    save_state(state_path, state)
    logger.info("Documentation upsert complete. Updated state: %s", state_path)


if __name__ == "__main__":
    main()
