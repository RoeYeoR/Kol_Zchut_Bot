"""The conversational RAG chain: condense → retrieve → rerank → generate.

Grounding rules (where hallucination is won or lost) — the answer must: 1) use ONLY the
numbered sources, 2) cite the source number(s), 3) say "לא נמצאה תשובה" when unsupported.

What makes it *conversational* (not stateless Q&A):
- Before retrieving, a follow-up is condensed against the conversation history into a
  standalone query — otherwise "וכמה מגיע לי?" retrieves nothing (see app/memory.py).
- Generation sees the recent turns (as real user/assistant messages) plus a rolling summary,
  so answers stay coherent across the chat without the prompt growing unbounded.
- The condensing + summarizing run on a small fast model (CONDENSE_MODEL); only the final
  grounded answer uses the strong model (ANSWER_MODEL).
"""
from __future__ import annotations

from anthropic import Anthropic

from app import memory
from app.config import ANSWER_MODEL, ANTHROPIC_API_KEY, CONDENSE_MODEL
from app.rerank import rerank
from app.retrieval import retrieve

_client: Anthropic | None = None


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        if not ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY is not set — needed for answer generation.")
        _client = Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


def _text(resp) -> str:
    return "".join(b.text for b in resp.content if b.type == "text")


SYSTEM = """אתה עוזר ידע בעברית בנושא זכויות עובדים ואזרחים. אתה עונה אך ורק על סמך המקורות הממוספרים שסופקו לך.

כללים:
- ענה בעברית ברורה ותמציתית, על בסיס המקורות בלבד. אל תשתמש בידע חיצוני ואל תמציא.
- ציטוט חובה: אחרי כל טענה צַיֵּן את מספר המקור בסוגריים מרובעים, למשל [1] או [2][3].
- אם המקורות אינם מכילים את התשובה, כתוב בדיוק: "לא נמצאה תשובה במאגר הידע." ואל תנחש.
- ייתכן שקיימת היסטוריית שיחה קודמת — השתמש בה רק להקשר (למשל למי מתייחסת השאלה), אך בַּסֵּס כל תשובה על המקורות הממוספרים של השאלה הנוכחית בלבד.
- אם השאלה כללית מדי, בקש הבהרה קצרה."""


def _condense(question: str, history: str) -> str:
    """Rewrite a follow-up into a standalone, searchable question using the history."""
    prompt = (
        f"היסטוריית שיחה:\n{history}\n\n"
        f"שאלת המשך: {question}\n\n"
        "נסח מחדש את שאלת ההמשך כשאלה עצמאית ומלאה שניתן להבין ולחפש לפיה במאגר ידע — "
        "פתור כינויי גוף והתייחסויות לפי ההיסטוריה. אם השאלה כבר עצמאית, החזר אותה כפי שהיא. "
        "החזר אך ורק את השאלה, בלי הסבר."
    )
    resp = _get_client().messages.create(
        model=CONDENSE_MODEL, max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    return _text(resp).strip() or question


def _summarize(prev_summary: str, overflow: list[memory.Turn]) -> str:
    """Fold turns that fell out of the verbatim window into a compact rolling summary."""
    convo = "\n".join(f"משתמש: {t.question}\nעוזרת: {t.answer}" for t in overflow)
    prompt = (
        (f"סיכום קודם:\n{prev_summary}\n\n" if prev_summary else "")
        + f"קטע שיחה נוסף לסיכום:\n{convo}\n\n"
        "עדכן לכדי סיכום תמציתי (2-3 משפטים) של ההקשר החשוב להמשך — פרטים על המשתמש ומצבו "
        "ומה נדון עד כה. החזר רק את הסיכום."
    )
    resp = _get_client().messages.create(
        model=CONDENSE_MODEL, max_tokens=250,
        messages=[{"role": "user", "content": prompt}],
    )
    return _text(resp).strip() or prev_summary


def _history_messages(session: memory.Session) -> list[dict]:
    """Recent turns as real user/assistant messages, with the rolling summary up front."""
    msgs: list[dict] = []
    if session.summary:
        msgs.append({"role": "user", "content": f"[הקשר מהשיחה עד כה]\n{session.summary}"})
        msgs.append({"role": "assistant", "content": "הבנתי, אמשיך מכאן."})
    for t in session.turns:
        msgs.append({"role": "user", "content": t.question})
        msgs.append({"role": "assistant", "content": t.answer})
    return msgs


def _format_sources(chunks: list[dict]) -> str:
    lines = []
    for i, c in enumerate(chunks, start=1):
        title = c["metadata"].get("title", "")
        lines.append(f"[{i}] מקור: {title}\n{c['text']}")
    return "\n\n".join(lines)


def _debug(search_q: str, retrieved: list[dict] | None, top: list[dict]) -> dict:
    return {
        "search_query": search_q,
        "retrieved": [
            {"title": c["metadata"].get("title"), "score": round(c["score"], 4), "text": c["text"]}
            for c in (retrieved or [])
        ],
        "reranked": [
            {
                "title": c["metadata"].get("title"),
                "rerank_score": round(c.get("rerank_score", 0.0), 4),
                "reranked": c.get("reranked", False),
                "text": c["text"],
            }
            for c in top
        ],
    }


def answer(
    question: str,
    session_id: str | None = None,
    reranked: list[dict] | None = None,
) -> dict:
    """Answer `question`. With a session_id the chain is multi-turn: it condenses the
    follow-up against history, then generates with that history in context. `reranked`
    lets a caller (the eval harness) pass chunks it already retrieved, to avoid repeat
    API calls — it also implies a single-shot, memory-free call."""
    session = memory.get(session_id) if session_id else None
    hist = memory.history_text(session) if session else ""

    # 1) condense a follow-up into a standalone search query (only when there's history)
    search_q = _condense(question, hist) if hist else question

    # 2) retrieve + rerank on the standalone query (or reuse chunks the caller passed)
    retrieved = retrieve(search_q) if reranked is None else None
    top = rerank(search_q, retrieved) if reranked is None else reranked

    if not top:
        out = {
            "answer": "לא נמצאה תשובה במאגר הידע.",
            "sources": [],
            "search_query": search_q,
            "debug": _debug(search_q, retrieved, []),
        }
        if session_id:
            memory.add_turn(session_id, question, out["answer"], summarizer=_summarize)
        return out

    # 3) generate — prior turns give conversational context; sources enforce grounding
    context = _format_sources(top)
    user = f"מקורות:\n\n{context}\n\nשאלה: {question}\n\nעני על סמך המקורות בלבד, עם ציטוטים."
    messages = (_history_messages(session) if session else []) + [{"role": "user", "content": user}]

    resp = _get_client().messages.create(
        model=ANSWER_MODEL, max_tokens=700, system=SYSTEM, messages=messages,
    )
    text = _text(resp).strip()

    sources = [
        {
            "n": i + 1,
            "title": c["metadata"].get("title", ""),
            "source_url": c["metadata"].get("source_url", ""),
            "chunk_id": c.get("metadata", {}).get("doc_id", "") + f"::{c['metadata'].get('chunk_index', '')}",
            "rerank_score": round(c.get("rerank_score", c.get("score", 0.0)), 4),
            "reranked": c.get("reranked", False),
        }
        for i, c in enumerate(top)
    ]

    # 4) persist the turn (folds overflow into the rolling summary when needed)
    if session_id:
        memory.add_turn(session_id, question, text, summarizer=_summarize)

    return {"answer": text, "sources": sources, "search_query": search_q, "debug": _debug(search_q, retrieved, top)}


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "כמה ימי הודעה מוקדמת מגיעים לעובד חודשי ותיק?"
    res = answer(q)
    print("שאלה:", q)
    print("\nתשובה:\n", res["answer"])
    print("\nמקורות:")
    for s in res["sources"]:
        print(f"  [{s['n']}] {s['title']}  (score={s['rerank_score']}, reranked={s['reranked']})")
