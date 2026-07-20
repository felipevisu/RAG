"""Persistence layer — pgvector does the storage, scoring, and sorting."""

import os

import psycopg

DB_URL = os.environ.get(
    "DB_URL", "host=localhost port=5433 dbname=demo3 user=postgres password=postgres"
)


def db():
    return psycopg.connect(DB_URL, autocommit=True)


def insert_post(content: str, category: str | None, embedding: str) -> dict:
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO posts (content, category, embedding) VALUES (%s, %s, %s) "
            "RETURNING id, content, category, embedding::text, created_at",
            (content, category, embedding),
        )
        id_, content, category, emb, created_at = cur.fetchone()
    return {"id": id_, "content": content, "category": category,
            "embedding": emb, "created_at": created_at}


def list_posts() -> list[dict]:
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, content, category, embedding::text, created_at "
            "FROM posts ORDER BY created_at DESC"
        )
        return [
            {"id": i, "content": c, "category": g, "embedding": e, "created_at": t}
            for i, c, g, e, t in cur.fetchall()
        ]


def delete_post(post_id: int) -> bool:
    with db() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM posts WHERE id = %s RETURNING id", (post_id,))
        return cur.fetchone() is not None


def search(query_embedding: str, limit: int = 5) -> list[tuple[str, str, float]]:
    """Top matches by cosine distance — the ORDER BY runs inside PostgreSQL."""
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT content, category, 1 - (embedding <=> %(q)s::vector) AS similarity
            FROM posts
            ORDER BY embedding <=> %(q)s::vector
            LIMIT %(limit)s
            """,
            {"q": query_embedding, "limit": limit},
        )
        return cur.fetchall()
