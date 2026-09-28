"""The RAG chain: retrieve -> rerank -> generate a grounded, cited Hebrew answer.

The generation prompt is where hallucination is won or lost. Three rules do the work:
1) answer ONLY from the numbered sources, 2) cite the source number(s) you used, and
3) if the sources don't contain the answer, say "לא נמצאה תשובה" instead of guessing.
For a rights/benefits assistant, a confident wrong answer is worse than "I don't know".
"""
from __future__ import annotations

from anthropic import Anthropic

from app.config import ANSWER_MODEL, ANTHROPIC_API_KEY
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


SYSTEM = """את עוזרת ידע בעברית בנושא זכויות עובדים ואזרחים. את עונה אך ורק על סמך המקורות הממוספרים שסופקו לך.

כללים:
- עני בעברית ברורה ותמציתית, על בסיס המקורות בלבד. אל תשתמשי בידע חיצוני ואל תמציאי.
- ציטוט חובה: אחרי כל טענה ציני את מספר המקור בסוגריים מרובעים, למשל [1] או [2][3].
- אם המקורות אינם מכילים את התשובה, כתבי בדיוק: "לא נמצאה תשובה במאגר הידע." ואל תנחשי.
- אם השאלה כללית מדי, בקשי הבהרה קצרה."""


def _format_sources(chunks: list[dict]) -> str:
    lines = []
    for i, c in enumerate(chunks, start=1):
        title = c["metadata"].get("title", "")
        lines.append(f"[{i}] מקור: {title}\n{c['text']}")
    return "\n\n".join(lines)


def answer(question: str) -> dict:
    retrieved = retrieve(question)
    top = rerank(question, retrieved)

    if not top:
        return {"answer": "לא נמצאה תשובה במאגר הידע.", "sources": [], "retrieved": retrieved}

    context = _format_sources(top)
    user = f"מקורות:\n\n{context}\n\nשאלה: {question}\n\nעני על סמך המקורות בלבד, עם ציטוטים."

    resp = _get_client().messages.create(
        model=ANSWER_MODEL,
        max_tokens=700,
        system=SYSTEM,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")

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
    return {"answer": text.strip(), "sources": sources, "retrieved": retrieved}


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "כמה ימי הודעה מוקדמת מגיעים לעובד חודשי ותיק?"
    res = answer(q)
    print("שאלה:", q)
    print("\nתשובה:\n", res["answer"])
    print("\nמקורות:")
    for s in res["sources"]:
        print(f"  [{s['n']}] {s['title']}  (score={s['rerank_score']}, reranked={s['reranked']})")
