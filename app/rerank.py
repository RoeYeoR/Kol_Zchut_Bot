"""Reranking: re-sort the retrieved chunks by TRUE relevance, keep the best few.

Why a second stage? The vector search compares two embeddings computed independently —
it never reads the question and a chunk *together*. A cross-encoder reranker does exactly
that: it takes (question, chunk) as one input and scores how well the chunk answers the
question. It's slower, so we only run it on the K candidates from retrieval, then keep the
top N. This is usually the single biggest quality lever in a RAG system: retrieval decides
what's *available*, reranking decides what the model actually *reads*.

We use Cohere Rerank (multilingual, good on Hebrew). If no COHERE_API_KEY is set, we fall
back to the vector-similarity order so the app still runs — but we tell you it's degraded.
"""
from __future__ import annotations

from app.config import COHERE_API_KEY, RERANK_MODEL, RERANK_N

_client = None


def _get_client():
    global _client
    if _client is None:
        import cohere

        _client = cohere.ClientV2(api_key=COHERE_API_KEY)
    return _client


def rerank(question: str, candidates: list[dict], n: int = RERANK_N) -> list[dict]:
    if not candidates:
        return []
    if not COHERE_API_KEY:
        # Fallback: no reranker available — trust the vector order (lower precision).
        for c in candidates:
            c["rerank_score"] = c.get("score", 0.0)
            c["reranked"] = False
        return candidates[:n]

    res = _get_client().rerank(
        model=RERANK_MODEL,
        query=question,
        documents=[c["text"] for c in candidates],
        top_n=n,
    )
    out: list[dict] = []
    for r in res.results:
        chunk = dict(candidates[r.index])
        chunk["rerank_score"] = float(r.relevance_score)
        chunk["reranked"] = True
        out.append(chunk)
    return out
