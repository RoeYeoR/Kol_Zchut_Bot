"""Conversation memory — turns the assistant from stateless Q&A into a real multi-turn chat.

Allowing follow-ups creates two problems, and this module owns the state for both:

1. Retrieval needs a STANDALONE query. A follow-up like "וכמה מגיע לי?" retrieves nothing on
   its own — its referent ("פיצויי פיטורים", "לעובד ותיק") lives in earlier turns. So before
   retrieving, rag.py condenses (history + follow-up) → one self-contained question.

2. The prompt must not grow without bound. We keep the last MAX_TURNS exchanges verbatim and
   fold anything older into a rolling summary, so cost and latency stay flat over a long chat.

For a POC this store is in-process (a dict keyed by session_id). Swap point: Redis or a DB
for persistence and horizontal scale — the interface (get / history_text / add_turn) is the
same, only the backing store changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

MAX_TURNS = 6  # keep the last 6 exchanges verbatim; older ones get summarized


@dataclass
class Turn:
    question: str
    answer: str


@dataclass
class Session:
    summary: str = ""
    turns: list[Turn] = field(default_factory=list)


_sessions: dict[str, Session] = {}


def get(session_id: str) -> Session:
    return _sessions.setdefault(session_id, Session())


def reset(session_id: str) -> None:
    _sessions.pop(session_id, None)


def history_text(session: Session) -> str:
    """Render memory as plain text: the rolling summary followed by recent verbatim turns."""
    parts: list[str] = []
    if session.summary:
        parts.append(f"סיכום השיחה עד כה: {session.summary}")
    for t in session.turns:
        parts.append(f"משתמש: {t.question}\nעוזרת: {t.answer}")
    return "\n\n".join(parts)


def add_turn(
    session_id: str,
    question: str,
    answer: str,
    summarizer: Callable[[str, list[Turn]], str] | None = None,
) -> None:
    """Append a turn. When we exceed MAX_TURNS, fold the overflow into the rolling summary."""
    s = get(session_id)
    s.turns.append(Turn(question, answer))
    if len(s.turns) > MAX_TURNS:
        overflow = s.turns[:-MAX_TURNS]
        s.turns = s.turns[-MAX_TURNS:]
        if summarizer is not None:
            s.summary = summarizer(s.summary, overflow)
