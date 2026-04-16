from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import httpx

from .db import db


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def notes_dir() -> Path:
    return Path(os.environ.get("NOTES_DIR", "notes")).resolve()


def ollama_host() -> str:
    return os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


def embed_model() -> str:
    return os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")


@dataclass(frozen=True)
class Chunk:
    content: str
    start: int
    end: int


def chunk_markdown(text: str, target_chars: int = 900) -> list[Chunk]:
    parts = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    chunks: list[Chunk] = []
    cursor = 0

    def _find_next_span(needle: str, start_at: int) -> tuple[int, int]:
        idx = text.find(needle, start_at)
        if idx == -1:
            return start_at, min(start_at + len(needle), len(text))
        return idx, idx + len(needle)

    buffer: list[str] = []
    buffer_start = 0
    for part in parts:
        if not buffer:
            buffer_start, _ = _find_next_span(part, cursor)

        if sum(len(x) for x in buffer) + len(part) + 2 <= target_chars:
            buffer.append(part)
            cursor = buffer_start + sum(len(x) for x in buffer) + 2 * (len(buffer) - 1)
            continue

        content = "\n\n".join(buffer).strip()
        start, end = _find_next_span(content, buffer_start)
        chunks.append(Chunk(content=content, start=start, end=end))

        buffer = [part]
        buffer_start, _ = _find_next_span(part, end)
        cursor = buffer_start

    if buffer:
        content = "\n\n".join(buffer).strip()
        start, end = _find_next_span(content, buffer_start)
        chunks.append(Chunk(content=content, start=start, end=end))

    return chunks


async def embed_text(text: str) -> list[float]:
    url = f"{ollama_host()}/api/embeddings"
    payload = {"model": embed_model(), "prompt": text}
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "Failed to call Ollama embeddings. Make sure Ollama is running "
            "and you ran: `ollama pull nomic-embed-text`"
        ) from e

    emb = data.get("embedding")
    if not isinstance(emb, list) or not emb:
        raise RuntimeError("Ollama returned no embedding vector.")
    return [float(x) for x in emb]


def _norm(vec: list[float]) -> float:
    return float(sum(x * x for x in vec) ** 0.5)


def iter_markdown_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*.md"):
        if p.is_file():
            yield p


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


ENTITY_RE = re.compile(r"(?:^|[\s(])([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+){0,3})(?=$|[\s).,;:])")
TAG_RE = re.compile(r"(?<!\w)([#@][a-zA-Z][\w-]{1,50})")


def extract_entities(text: str) -> list[str]:
    found: list[str] = []
    for m in TAG_RE.finditer(text):
        found.append(m.group(1))
    for m in ENTITY_RE.finditer(text):
        cand = m.group(1).strip()
        if len(cand) < 3:
            continue
        if cand.lower() in {"the", "and", "for", "with"}:
            continue
        found.append(cand)

    # de-dupe, preserve rough order
    seen = set()
    out: list[str] = []
    for e in found:
        key = e.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


async def reindex() -> dict:
    root = notes_dir()
    if not root.exists():
        raise RuntimeError(f"Notes dir does not exist: {root}")

    indexed_docs = 0
    indexed_chunks = 0
    skipped_docs = 0

    with db() as conn:
        # Clear graph each reindex (simple + deterministic).
        conn.execute("DELETE FROM edges;")
        conn.execute("DELETE FROM entity_mentions;")
        conn.execute("DELETE FROM entities;")

    for path in iter_markdown_files(root):
        text = path.read_text(encoding="utf-8")
        digest = sha256_text(text)

        with db() as conn:
            row = conn.execute(
                "SELECT id, sha256 FROM documents WHERE path = ?",
                (str(path),),
            ).fetchone()

        if row and row["sha256"] == digest:
            skipped_docs += 1
            continue

        with db() as conn:
            if row:
                doc_id = int(row["id"])
                conn.execute("UPDATE documents SET sha256 = ?, updated_at = ? WHERE id = ?", (digest, _utc_now_iso(), doc_id))
                conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
            else:
                cur = conn.execute(
                    "INSERT INTO documents(path, sha256, updated_at) VALUES (?, ?, ?)",
                    (str(path), digest, _utc_now_iso()),
                )
                doc_id = int(cur.lastrowid)

        chunks = chunk_markdown(text)
        indexed_docs += 1

        # Insert chunks first (sync), then embed async one-by-one (simpler).
        chunk_rows: list[tuple[int, int, str, int, int]] = []
        for i, ch in enumerate(chunks):
            chunk_rows.append((doc_id, i, ch.content, ch.start, ch.end))

        with db() as conn:
            conn.executemany(
                "INSERT INTO chunks(document_id, chunk_index, content, start_offset, end_offset) VALUES (?, ?, ?, ?, ?)",
                chunk_rows,
            )

        with db() as conn:
            chunk_db_rows = conn.execute(
                "SELECT id, content FROM chunks WHERE document_id = ? ORDER BY chunk_index",
                (doc_id,),
            ).fetchall()

        for r in chunk_db_rows:
            vec = await embed_text(r["content"])
            n = _norm(vec)
            with db() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO embeddings(chunk_id, model, vector_json, norm) VALUES (?, ?, ?, ?)",
                    (int(r["id"]), embed_model(), json.dumps(vec), n),
                )
            indexed_chunks += 1

        # Entities + edges per document (co-occur).
        ents = extract_entities(text)
        ent_ids: list[int] = []
        with db() as conn:
            for e in ents:
                conn.execute("INSERT OR IGNORE INTO entities(name) VALUES (?)", (e,))
                eid = conn.execute("SELECT id FROM entities WHERE name = ?", (e,)).fetchone()["id"]
                ent_ids.append(int(eid))

            # Mentions count (very rough: substring count)
            for e, eid in zip(ents, ent_ids, strict=False):
                mentions = max(1, len(re.findall(re.escape(e), text)))
                conn.execute(
                    "INSERT INTO entity_mentions(entity_id, document_id, mentions) VALUES (?, ?, ?) "
                    "ON CONFLICT(entity_id, document_id) DO UPDATE SET mentions = mentions + excluded.mentions",
                    (eid, doc_id, mentions),
                )

            # Undirected co-occurrence edges stored as (min,max).
            for i in range(len(ent_ids)):
                for j in range(i + 1, len(ent_ids)):
                    a, b = ent_ids[i], ent_ids[j]
                    src, dst = (a, b) if a < b else (b, a)
                    conn.execute(
                        "INSERT INTO edges(src_entity_id, dst_entity_id, weight) VALUES (?, ?, 1) "
                        "ON CONFLICT(src_entity_id, dst_entity_id) DO UPDATE SET weight = weight + 1",
                        (src, dst),
                    )

    return {
        "notes_dir": str(root),
        "indexed_docs": indexed_docs,
        "indexed_chunks": indexed_chunks,
        "skipped_docs": skipped_docs,
        "embed_model": embed_model(),
    }


def cosine_similarity(a: list[float], a_norm: float, b: list[float], b_norm: float) -> float:
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    dot = 0.0
    for x, y in zip(a, b, strict=False):
        dot += float(x) * float(y)
    return dot / (a_norm * b_norm)

