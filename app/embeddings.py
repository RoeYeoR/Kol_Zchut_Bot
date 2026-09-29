"""Embeddings: turn text into vectors.

The same model must embed both the documents (at ingest) and the query (at search) —
they have to live in the same vector space for cosine similarity to mean anything.
We use Cohere embed-multilingual-v3.0: strong on Hebrew, and it lets the whole app run
on a single Cohere key (embeddings + rerank) plus Claude for generation — no OpenAI needed.

Cohere embeddings are asymmetric: documents are embedded with input_type="search_document"
and queries with "search_query". Matching the type on each side measurably improves
retrieval, so we expose it per call instead of embedding everything the same way.

Swap point: for a no-API-key / on-prem setup, replace this with a local multilingual model
(e.g. intfloat/multilingual-e5-large via sentence-transformers).
"""
from __future__ import annotations

from app.config import COHERE_API_KEY, EMBEDDING_MODEL
from app.retry import with_backoff

_client = None


def _get_client():
    global _client
    if _client is None:
        if not COHERE_API_KEY:
            raise RuntimeError("COHERE_API_KEY is not set — needed for embeddings.")
        import cohere

        _client = cohere.ClientV2(api_key=COHERE_API_KEY)
    return _client


def _embed(texts: list[str], input_type: str) -> list[list[float]]:
    if not texts:
        return []
    resp = with_backoff(
        lambda: _get_client().embed(
            texts=texts,
            model=EMBEDDING_MODEL,
            input_type=input_type,
            embedding_types=["float"],
        )
    )
    return resp.embeddings.float


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed documents for storage (ingest). Batching keeps ingest fast and cheap."""
    return _embed(texts, "search_document")


def embed_query(text: str) -> list[float]:
    """Embed a search query — asymmetric to document embedding on purpose."""
    return _embed([text], "search_query")[0]