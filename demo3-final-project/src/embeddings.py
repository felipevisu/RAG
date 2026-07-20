"""Transformer layer — turns text into vectors.

The sentence transformer (BAAI/bge-base-en-v1.5) is the "embedding model":
a neural network that maps a sentence to a 768-dim vector where similar
meanings land close together. It runs locally, no API involved.
"""

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-base-en-v1.5")  # 768-dim embeddings

# bge models retrieve better when short queries carry this instruction prefix
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def embed_passage(text: str) -> str:
    """Embed stored content. Returns pgvector literal like '[0.1, -0.2, ...]'."""
    return str(model.encode(text).tolist())


def embed_query(text: str) -> str:
    """Embed a search query (with the bge instruction prefix)."""
    return str(model.encode(QUERY_PREFIX + text).tolist())
