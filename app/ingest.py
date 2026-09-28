"""Ingestion pipeline (run once, offline): corpus -> chunk -> embed -> vector store.

Separating ingest from query is a core RAG principle: the expensive work (chunking +
embedding the whole corpus) happens ahead of time, so each user query only has to embed
one short question and do a fast nearest-neighbour lookup.
"""
from __future__ import annotations

from app.chunking import load_chunks
from app.embeddings import embed_texts
from app.vectorstore import count, reset, upsert


def ingest() -> None:
    print("Loading + chunking corpus…")
    chunks = load_chunks()
    print(f"  {len(chunks)} chunks")

    print("Embedding…")
    vectors = embed_texts([c.text for c in chunks])

    print("Writing to vector store…")
    reset()  # rebuild from scratch so re-ingest is idempotent
    upsert(
        ids=[c.id for c in chunks],
        texts=[c.text for c in chunks],
        embeddings=vectors,
        metadatas=[c.metadata for c in chunks],
    )
    print(f"Done. Vector store now holds {count()} chunks.")


if __name__ == "__main__":
    ingest()
