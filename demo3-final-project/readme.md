# Demo 3 — Final project: RAG feed + Claude chatbot

A Twitter-style feed where every post is embedded and persisted in PostgreSQL (pgvector), plus a chatbot that answers questions using your posts as a knowledge base via Claude **tool_use**.

## Architecture

```
Browser (Feed / Chat tabs)
   │
   ▼
FastAPI (app.py)
   ├── POST /api/posts  → sentence-transformers embeds → INSERT into pgvector
   ├── GET  /api/posts  → posts + their stored vectors (shown in the UI)
   └── POST /api/chat   → Claude (claude-opus-4-8) with a search_knowledge_base
                          tool; Claude decides when to search, PostgreSQL ranks
                          by cosine distance, Claude answers from the results
```

The chat endpoint runs the classic tool_use loop: call Claude → if `stop_reason == "tool_use"`, run the SQL similarity search and send the results back as a `tool_result` → repeat until Claude produces a final text answer.

## Run

Everything runs on docker compose (database + backend + UI):

```bash
# put your key in .env (needed for the Chat tab):
#   ANTHROPIC_API_KEY=sk-ant-...
docker compose up -d --build
```

Open http://localhost:8000

> First startup downloads the sentence-transformers model (cached in a volume afterwards). To develop outside docker: keep only the `db` service running and start `../venv/bin/uvicorn app:app --reload` from this folder.

- **Feed tab** — write posts; each shows its stored 384-dim vector (click to expand).
- **Chat tab** — ask questions; the UI shows which searches Claude performed before answering.
