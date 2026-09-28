"""Load corpus documents, parse their metadata, and split them into overlapping chunks.

A "chunk" is the unit we embed and retrieve. Good chunking is the quiet lever most
people underrate: chunks that are too large dilute the embedding (the vector averages
several topics, so retrieval gets noisy); too small and a chunk loses the context needed
to answer. We split on paragraph boundaries and carry an overlap so a fact that straddles
a boundary still appears whole in at least one chunk.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from app.config import CHUNK_OVERLAP, CHUNK_SIZE, CORPUS_DIR


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


_FRONT_MATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _parse_doc(path: Path) -> tuple[dict, str]:
    """Split a corpus file into its metadata header and its body text."""
    raw = path.read_text(encoding="utf-8")
    meta: dict = {"doc_id": path.stem, "source_url": "", "title": path.stem, "category": ""}
    m = _FRONT_MATTER.match(raw)
    body = raw
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                key, _, val = line.partition(":")
                meta[key.strip()] = val.strip()
        body = raw[m.end():]
    return meta, body.strip()


def _split_text(text: str) -> list[str]:
    """Paragraph-aware splitter with character overlap.

    We accumulate whole paragraphs until adding the next one would exceed CHUNK_SIZE,
    then start a new chunk that begins with the last CHUNK_OVERLAP characters of the
    previous one (so context carries across the seam).
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > CHUNK_SIZE:
            chunks.append(current.strip())
            tail = current[-CHUNK_OVERLAP:]
            current = tail + "\n\n" + para
        else:
            current = para if not current else current + "\n\n" + para
    if current.strip():
        chunks.append(current.strip())
    return chunks


def load_chunks() -> list[Chunk]:
    """Read every doc in the corpus and return the flat list of chunks to index."""
    chunks: list[Chunk] = []
    for path in sorted(CORPUS_DIR.glob("*.md")):
        meta, body = _parse_doc(path)
        for i, piece in enumerate(_split_text(body)):
            chunks.append(
                Chunk(
                    id=f"{meta['doc_id']}::{i}",
                    text=piece,
                    metadata={**meta, "chunk_index": i},
                )
            )
    return chunks


if __name__ == "__main__":
    cs = load_chunks()
    print(f"{len(cs)} chunks from {len(list(CORPUS_DIR.glob('*.md')))} docs")
    for c in cs[:3]:
        print(f"\n--- {c.id} ({c.metadata['title']}) ---\n{c.text[:200]}…")
