"""Retrieval: embed the question, pull the top-K most similar chunks from the vector store.

This stage optimizes for RECALL — cast a wide net (K is generous). Precision comes later,
in reranking. Dense (embedding) search is great at meaning/paraphrase but can miss exact
tokens (a specific number or name); a production system adds a keyword (BM25) leg and
fuses the two — that's "hybrid retrieval", noted as a swap point in the README.
"""
from __future__ import annotations

from app.config import RETRIEVE_K
from app.embeddings import embed_query
from app.vectorstore import query


def retrieve(question: str, k: int = RETRIEVE_K) -> list[dict]:
    q_vec = embed_query(question)
    return query(q_vec, k)
