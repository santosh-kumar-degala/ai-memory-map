from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .db import db, init_db
from .graph import fetch_graph
from .indexer import cosine_similarity, embed_text, embed_model, notes_dir, reindex


app = FastAPI(title="AI Memory Map", version="0.1.0")


STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.on_event("startup")
def _startup() -> None:
    init_db()


@app.get("/")
def root() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "notes_dir": str(notes_dir()),
        "embed_model": embed_model(),
        "db_path": os.environ.get("DB_PATH", "memory_map.sqlite"),
    }


@app.post("/api/reindex")
async def api_reindex() -> dict[str, Any]:
    try:
        return await reindex()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/graph")
def api_graph(limit_nodes: int = 200, min_edge_weight: int = 1) -> dict[str, Any]:
    return fetch_graph(limit_nodes=limit_nodes, min_edge_weight=min_edge_weight)


@app.get("/api/search")
async def api_search(q: str, k: int = 8) -> dict[str, Any]:
    q = (q or "").strip()
    if not q:
        return {"query": q, "results": []}

    try:
        q_vec = await embed_text(q)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e)) from e

    q_norm = float(sum(x * x for x in q_vec) ** 0.5)
    rows: list[dict[str, Any]] = []

    with db() as conn:
        chunk_rows = conn.execute(
            """
            SELECT
              c.id AS chunk_id,
              c.content,
              c.chunk_index,
              d.path AS doc_path,
              e.vector_json,
              e.norm
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            JOIN embeddings e ON e.chunk_id = c.id
            """
        ).fetchall()

    for r in chunk_rows:
        vec = json.loads(r["vector_json"])
        score = cosine_similarity(q_vec, q_norm, vec, float(r["norm"]))
        rows.append(
            {
                "score": score,
                "doc_path": str(r["doc_path"]),
                "chunk_index": int(r["chunk_index"]),
                "content": str(r["content"]),
            }
        )

    rows.sort(key=lambda x: x["score"], reverse=True)
    top = rows[: max(1, min(int(k), 50))]

    return {
        "query": q,
        "k": k,
        "embed_model": embed_model(),
        "results": top,
    }

