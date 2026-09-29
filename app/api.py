"""FastAPI service.

- POST /ask     -> grounded, cited answer (the product endpoint)
- POST /search  -> raw retrieval + rerank scores (the debug/eval endpoint used for
                   root-cause analysis: was a wrong answer a retrieval miss or a
                   generation miss?)
- GET  /        -> the minimal query + debug UI
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app import memory
from app.rag import answer
from app.rerank import rerank
from app.retrieval import retrieve
from app.vectorstore import count

app = FastAPI(title="זכותון — Hebrew Rights RAG")

UI = (Path(__file__).resolve().parent.parent / "ui" / "index.html").read_text(encoding="utf-8")


class Query(BaseModel):
    question: str
    session_id: str | None = None  # pass one to make the chat multi-turn (memory-aware)


class Session(BaseModel):
    session_id: str


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return UI


@app.get("/health")
def health() -> dict:
    return {"chunks_indexed": count()}


@app.post("/ask")
def ask(q: Query) -> dict:
    return answer(q.question, session_id=q.session_id)


@app.post("/reset")
def reset(s: Session) -> dict:
    """Start a fresh conversation — drops this session's memory (history + summary)."""
    memory.reset(s.session_id)
    return {"ok": True}


@app.post("/search")
def search(q: Query) -> dict:
    """Retrieval + rerank only — no generation. Shows exactly what the model would see."""
    retrieved = retrieve(q.question)
    reranked = rerank(q.question, retrieved)
    return {
        "retrieved": [
            {"title": c["metadata"].get("title"), "score": round(c["score"], 4), "text": c["text"]}
            for c in retrieved
        ],
        "reranked": [
            {
                "title": c["metadata"].get("title"),
                "rerank_score": round(c.get("rerank_score", 0.0), 4),
                "reranked": c.get("reranked", False),
                "text": c["text"],
            }
            for c in reranked
        ],
    }
