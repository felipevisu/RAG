"""AI layer — Claude (the LLM) answering with tool_use over the knowledge base.

This is the generative side: Claude decides *when* to search, the transformer
(embeddings.py) turns the query into a vector, and PostgreSQL (database.py)
ranks the results. The tool_use loop below repeats until Claude stops asking
for searches and writes its final answer.
"""

import anthropic
from fastapi import HTTPException

from . import database, embeddings

claude = anthropic.Anthropic()

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
    "posts and imported documents (each entry is tagged with a category, e.g. "
    "a policy document name). Use the search_knowledge_base tool to look up "
    "relevant entries before answering questions about the user's life, notes, "
    "or documents. Base your answers on the retrieved entries, mention which "
    "category they came from, and say so when nothing relevant is found."
)


def search_knowledge_base(query: str) -> str:
    rows = database.search(embeddings.embed_query(query))
    if not rows:
        return "The knowledge base is empty."
    return "\n".join(
        f"[{category or 'post'} | similarity {sim:.3f}] {content}"
        for content, category, sim in rows
    )


def chat(messages: list) -> dict:
    """Run the tool_use loop; returns the final reply and the searches made."""
    messages = list(messages)
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
                502,
                "Anthropic auth failed — set ANTHROPIC_API_KEY in .env and "
                "restart: docker compose up -d",
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
