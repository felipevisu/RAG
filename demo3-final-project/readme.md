# Demo 3 — Final project: RAG feed + Claude chatbot

A Twitter-style feed where every post is embedded and persisted in PostgreSQL (pgvector), plus a chatbot that answers questions using your posts as a knowledge base via Claude **tool_use**.

## Architecture — one file per concern

| File | Layer | What it is |
|---|---|---|
| `src/app.py` | **API** | FastAPI HTTP routes + serves the UI. No ML, no SQL, no Claude. |
| `src/embeddings.py` | **Transformer** | The sentence transformer (`BAAI/bge-base-en-v1.5`): a local neural network mapping text → 768-dim vector. Not generative — it only encodes meaning. |
| `src/database.py` | **Persistence** | PostgreSQL + pgvector: stores the vectors and does the scoring/sorting (`ORDER BY embedding <=> query`). |
| `src/assistant.py` | **AI (LLM)** | Claude (`claude-opus-4-8`): the generative model. Decides *when* to search via tool_use, reads the results, writes the answer. |
| `ingest.py` | script | Splits `documents/*.md` into `##` sections and posts them with a category. |

```
Browser (Feed / Chat tabs)
   │
   ▼
src/app.py (API)
   ├── POST /api/posts ──▶ embeddings.py ──▶ database.py (INSERT)
   ├── GET  /api/posts ──▶ database.py
   ├── DELETE /api/posts/{id} ──▶ database.py
   └── POST /api/chat ──▶ assistant.py (Claude tool_use loop)
                              └─ search tool ──▶ embeddings.py ──▶ database.py (rank)
```

The chat runs the classic tool_use loop: call Claude → if `stop_reason == "tool_use"`, run the SQL similarity search and send the results back as a `tool_result` → repeat until Claude produces a final text answer.

## Run

Database on docker (shared instance at the repo root), app on localhost:

```bash
docker compose up -d      # from the REPO ROOT — postgres serving demo2 + demo3 (port 5433)
cd demo3-final-project
# put your key in .env (needed for the Chat tab):
#   ANTHROPIC_API_KEY=sk-ant-...
../venv/bin/uvicorn src.app:app --reload --port 8080  # backend + UI
python ingest.py          # load documents/*.md into the knowledge base
```

Open http://localhost:8080

> Dependencies are in the root `requirements.txt`. `.env` is loaded automatically by `src/__init__.py` (a few lines of stdlib — no python-dotenv). This app uses the `demo3` database inside the shared instance; demo2 uses `demo2`. First startup downloads the sentence-transformers model (~440MB, cached afterwards).

- **Feed tab** — write posts; each shows its stored 768-dim vector (click to expand). Embeddings come from `BAAI/bge-base-en-v1.5`.
- **Chat tab** — ask questions; the UI shows which searches Claude performed before answering.
