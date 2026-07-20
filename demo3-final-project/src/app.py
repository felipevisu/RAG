"""API layer — FastAPI routes only. No ML, no SQL, no Claude here.

Layers:
    embeddings.py  → transformer (text → vector)
    database.py    → persistence (pgvector stores, scores, sorts)
    assistant.py   → AI (Claude + tool_use loop)
    app.py         → API (HTTP in/out) + serves the UI
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import assistant, database, embeddings

app = FastAPI(title="RAG Feed")


class PostIn(BaseModel):
    content: str
    category: str | None = None


class ChatIn(BaseModel):
    messages: list  # [{role, content}] — plain text history from the UI


@app.post("/api/posts")
def create_post(post: PostIn):
    return database.insert_post(
        post.content, post.category, embeddings.embed_passage(post.content)
    )


@app.get("/api/posts")
def list_posts():
    return database.list_posts()


@app.delete("/api/posts/{post_id}")
def delete_post(post_id: int):
    if not database.delete_post(post_id):
        raise HTTPException(404, "post not found")
    return {"deleted": post_id}


@app.post("/api/chat")
def chat(body: ChatIn):
    return assistant.chat(body.messages)


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")
