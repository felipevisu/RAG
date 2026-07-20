"""Demo 3 — Twitter-style feed persisted in pgvector + Claude chatbot over it.

Run:  ANTHROPIC_API_KEY=sk-ant-... docker compose up -d --build
"""

import os

import anthropic
import psycopg
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

DB_URL = os.environ.get(
    "DB_URL", "host=localhost port=5434 dbname=rag user=postgres password=postgres"
)

model = SentenceTransformer("all-MiniLM-L6-v2")
claude = anthropic.Anthropic()
app = FastAPI(title="RAG Feed")


def db():
    return psycopg.connect(DB_URL, autocommit=True)


# ---------- Feed ----------

class PostIn(BaseModel):
    content: str


@app.post("/api/posts")
def create_post(post: PostIn):
    embedding = model.encode(post.content).tolist()
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO posts (content, embedding) VALUES (%s, %s) "
            "RETURNING id, content, embedding::text, created_at",
            (post.content, str(embedding)),
        )
        id_, content, emb, created_at = cur.fetchone()
    return {"id": id_, "content": content, "embedding": emb, "created_at": created_at}


@app.get("/api/posts")
def list_posts():
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, content, embedding::text, created_at "
            "FROM posts ORDER BY created_at DESC"
        )
        return [
            {"id": i, "content": c, "embedding": e, "created_at": t}
            for i, c, e, t in cur.fetchall()
        ]


# ---------- Chat (Claude tool_use over the knowledge base) ----------

SEARCH_TOOL = {
    "name": "search_knowledge_base",
    "description": (
        "Search the user's personal knowledge base (their posted notes) by "
        "semantic similarity. Call this whenever the answer may depend on "
        "something the user has posted."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for"}
        },
        "required": ["query"],
    },
}

SYSTEM = (
    "You are a personal assistant. The user keeps a knowledge base of short "
    "posts. Use the search_knowledge_base tool to look up relevant posts "
    "before answering questions about the user's life, notes, or opinions. "
    "Base your answers on the retrieved posts and say so when nothing "
    "relevant is found."
)


def search_knowledge_base(query: str) -> str:
    query_embedding = str(model.encode(query).tolist())
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT content, 1 - (embedding <=> %(q)s::vector) AS similarity
            FROM posts
            ORDER BY embedding <=> %(q)s::vector
            LIMIT 5
            """,
            {"q": query_embedding},
        )
        rows = cur.fetchall()
    if not rows:
        return "The knowledge base is empty."
    return "\n".join(f"[similarity {sim:.3f}] {content}" for content, sim in rows)


class ChatIn(BaseModel):
    messages: list  # [{role, content}] — plain text history from the UI


@app.post("/api/chat")
def chat(body: ChatIn):
    messages = list(body.messages)
    searches = []

    while True:
        try:
            response = claude.messages.create(
                model="claude-opus-4-8",
                max_tokens=16000,
                thinking={"type": "adaptive"},
                system=SYSTEM,
                tools=[SEARCH_TOOL],
                messages=messages,
            )
        except (anthropic.AuthenticationError, TypeError):
            # TypeError = SDK found no credentials at all (key unset/empty)
            raise HTTPException(
                502, "Anthropic auth failed — set ANTHROPIC_API_KEY and restart: "
                "ANTHROPIC_API_KEY=sk-ant-... docker compose up -d"
            )
        except anthropic.APIStatusError as e:
            raise HTTPException(502, f"Anthropic API error {e.status_code}: {e.message}")

        if response.stop_reason != "tool_use":
            break

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                searches.append(block.input["query"])
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": search_knowledge_base(block.input["query"]),
                })
        messages.append({"role": "user", "content": tool_results})

    reply = next((b.text for b in response.content if b.type == "text"), "")
    return {"reply": reply, "searches": searches}


# ---------- UI ----------

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")
