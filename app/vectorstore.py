"""Vector store: Qdrant in local (embedded) mode — a real vector DB with no server and
no C++ compiler needed (unlike Chroma's hnswlib). Each chunk's vector is stored with its
text + metadata as the payload, so a search result already carries everything needed to
cite the source. Swap point: point QdrantClient at a running Qdrant / pgvector for scale.
"""
from __future__ import annotations

import atexit

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.config import COLLECTION, STORE_DIR

_client: QdrantClient | None = None


def _c() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(path=str(STORE_DIR))
    return _client


@atexit.register
def _close() -> None:
    # Close the local client while the interpreter is still alive. Otherwise Qdrant's
    # own __del__ runs during shutdown (when sys.meta_path is gone) and prints a noisy,
    # harmless "Python is likely shutting down" ImportError traceback.
    global _client
    if _client is not None:
        try:
            _client.close()
        except Exception:
            pass
        _client = None


def reset() -> None:
    c = _c()
    if c.collection_exists(COLLECTION):
        c.delete_collection(COLLECTION)


def upsert(ids: list[str], texts: list[str], embeddings: list[list[float]], metadatas: list[dict]) -> None:
    c = _c()
    if not c.collection_exists(COLLECTION):
        c.create_collection(
            COLLECTION,
            vectors_config=VectorParams(size=len(embeddings[0]), distance=Distance.COSINE),
        )
    points = [
        PointStruct(id=i, vector=emb, payload={**meta, "text": text, "chunk_id": cid})
        for i, (cid, text, emb, meta) in enumerate(zip(ids, texts, embeddings, metadatas))
    ]
    c.upsert(collection_name=COLLECTION, points=points)


def query(embedding: list[float], k: int) -> list[dict]:
    c = _c()
    res = c.query_points(collection_name=COLLECTION, query=embedding, limit=k, with_payload=True)
    out: list[dict] = []
    for p in res.points:
        payload = dict(p.payload or {})
        text = payload.pop("text", "")
        out.append({"text": text, "metadata": payload, "score": float(p.score)})
    return out


def count() -> int:
    c = _c()
    return c.count(collection_name=COLLECTION).count if c.collection_exists(COLLECTION) else 0