from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path


def get_db_path() -> Path:
    return Path(os.environ.get("DB_PATH", "memory_map.sqlite")).resolve()


def connect() -> sqlite3.Connection:
    db_path = get_db_path()
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


@contextmanager
def db() -> sqlite3.Connection:
    conn = connect()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
              id INTEGER PRIMARY KEY,
              path TEXT NOT NULL UNIQUE,
              sha256 TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chunks (
              id INTEGER PRIMARY KEY,
              document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
              chunk_index INTEGER NOT NULL,
              content TEXT NOT NULL,
              start_offset INTEGER NOT NULL,
              end_offset INTEGER NOT NULL,
              UNIQUE(document_id, chunk_index)
            );

            CREATE TABLE IF NOT EXISTS embeddings (
              chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
              model TEXT NOT NULL,
              vector_json TEXT NOT NULL,
              norm REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS entities (
              id INTEGER PRIMARY KEY,
              name TEXT NOT NULL UNIQUE
            );

            CREATE TABLE IF NOT EXISTS entity_mentions (
              entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
              document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
              mentions INTEGER NOT NULL,
              PRIMARY KEY(entity_id, document_id)
            );

            CREATE TABLE IF NOT EXISTS edges (
              src_entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
              dst_entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
              weight INTEGER NOT NULL,
              PRIMARY KEY(src_entity_id, dst_entity_id)
            );
            """
        )

