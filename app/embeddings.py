"""Embeddings: turn text into vectors.

The same model must embed both the documents (at ingest) and the query (at search) —
they have to live in the same vector space for cosine similarity to mean anything.
We use OpenAI text-embedding-3-small: multilingual (so Hebrew works), cheap, low-latency.

Swap point: for a no-API-key / on-prem setup, replace `embed_texts` with a local
multilingual model (e.g. intfloat/multilingual-e5-large via sentence-transformers).
"""
from __future__ import annotations

from openai import OpenAI

from app.config import EMBEDDING_MODEL, OPENAI_API_KEY

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set — needed for embeddings.")
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. Batching keeps ingest fast and cheap."""
    if not texts:
        return []
    resp = _get_client().embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in resp.data]


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
