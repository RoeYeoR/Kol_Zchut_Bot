"""Vector store: persist chunk vectors + metadata and do approximate-nearest-neighbour search.

We use Chroma (on-disk, in-process — zero infra) so the POC runs with a single `pip install`.
Every vector stores its chunk text and metadata alongside it, so a search result already
carries everything we need to cite the source.

Swap point: for scale + query-time metadata filtering, move to pgvector / Qdrant / Pinecone.
The interface below (upsert / query) is deliberately small so the backend is replaceable.
"""
from __future__ import annotations

import chromadb

from app.config import CHROMA_DIR, COLLECTION


def _collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # cosine distance is the right metric for text embeddings
    return client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def reset() -> None:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass


def upsert(ids: list[str], texts: list[str], embeddings: list[list[float]], metadatas: list[dict]) -> None:
    _collection().upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)


def query(embedding: list[float], k: int) -> list[dict]:
    """Return the k nearest chunks as dicts with text, metadata, and a 0..1 similarity score."""
    res = _collection().query(query_embeddings=[embedding], n_results=k)
    out: list[dict] = []
    for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        out.append({"text": text, "metadata": meta, "score": 1.0 - float(dist)})  # cosine dist -> similarity
    return out


def count() -> int:
    return _collection().count()
