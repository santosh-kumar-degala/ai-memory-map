# AI Memory Map (local-first)

This is a tiny “personal knowledge graph + semantic search” starter you can grow into a unique AI project.

It:
- indexes Markdown notes from `notes/`
- generates **local embeddings via Ollama**
- stores docs/chunks/embeddings/entities/edges in **SQLite**
- serves a small web UI to search + visualize relationships

## Prereqs

- Python 3.10+
- (Recommended) Ollama installed and running

Install Ollama and pull an embedding model:

```bash
ollama pull nomic-embed-text
```

## Run

```bash
cd ai-memory-map
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`.

## Notes folder

Put `.md` files in `notes/`. Then hit **Reindex** in the UI (or call `POST /api/reindex`).

## Config

Environment variables:
- `NOTES_DIR` (default: `notes`)
- `DB_PATH` (default: `memory_map.sqlite`)
- `OLLAMA_HOST` (default: `http://localhost:11434`)
- `OLLAMA_EMBED_MODEL` (default: `nomic-embed-text`)

## Next ideas (to make it “wow”)

- Better entity extraction (spaCy / GLiNER) and typed edges (DECIDED, OWNS, DEPENDS_ON)
- Timeline view: decisions and tasks over time
- “Ask the graph”: cite source chunks when answering
- Watch mode: auto-reindex on file changes
