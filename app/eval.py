"""Evaluation harness — the answer to "how do you know quality actually improved?"

We measure the two failure modes separately, because a wrong final answer is either:
  (a) a RETRIEVAL miss — the right chunk was never fetched, or
  (b) a GENERATION miss — the right chunk was there but the model ignored/misread it.
Only measuring the final answer hides which one you have. So we report:

  Retrieval   : hit@K, hit@N (after rerank), MRR   — deterministic, from doc ids
  Generation  : answer correctness via an LLM judge — grounded + matches the expected fact

Run it before and after any change (chunk size, K/N, add reranking, swap embeddings) and
watch the numbers move. That before/after delta is the whole point.
"""
from __future__ import annotations

import json

from anthropic import Anthropic

from app.config import ANSWER_MODEL, ANTHROPIC_API_KEY, EVAL_FILE, RERANK_N, RETRIEVE_K
from app.rag import answer
from app.retrieval import retrieve
from app.rerank import rerank

_judge = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None


def _load() -> list[dict]:
    return [json.loads(line) for line in EVAL_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]


def _doc_ids(chunks: list[dict]) -> list[str]:
    return [c["metadata"].get("doc_id") for c in chunks]


def _first_relevant_rank(chunks: list[dict], relevant: list[str]) -> int | None:
    for rank, doc_id in enumerate(_doc_ids(chunks), start=1):
        if doc_id in relevant:
            return rank
    return None


def _judge_answer(question: str, model_answer: str, expected: str) -> bool:
    """LLM-as-judge: is the answer correct and faithful to the expected fact?"""
    if _judge is None:
        return False
    prompt = (
        f"שאלה: {question}\n"
        f"עובדה מצופה: {expected}\n"
        f"תשובת המערכת: {model_answer}\n\n"
        "האם תשובת המערכת נכונה ותואמת לעובדה המצופה (כולל המקרה שבו התשובה הנכונה היא "
        "שאין מידע במאגר)? ענה במילה אחת בלבד: כן / לא."
    )
    resp = _judge.messages.create(
        model=ANSWER_MODEL, max_tokens=5,
        messages=[{"role": "user", "content": prompt}],
    )
    verdict = "".join(b.text for b in resp.content if b.type == "text").strip()
    return verdict.startswith("כן")


def run() -> None:
    rows = _load()
    hit_k = hit_n = correct = 0
    mrr = 0.0

    print(f"Evaluating {len(rows)} questions (K={RETRIEVE_K}, N={RERANK_N})\n")
    for r in rows:
        q, relevant, expected = r["question"], r["relevant_doc_ids"], r["expected"]
        retrieved = retrieve(q)
        reranked = rerank(q, retrieved)

        # --- retrieval metrics ---
        if not relevant:
            # out-of-scope question: "retrieval" isn't the point, only the refusal is.
            in_k = in_n = True
            rank = None
        else:
            in_k = any(d in relevant for d in _doc_ids(retrieved))
            in_n = any(d in relevant for d in _doc_ids(reranked))
            rank = _first_relevant_rank(reranked, relevant)
        hit_k += int(in_k)
        hit_n += int(in_n)
        mrr += (1.0 / rank) if rank else (0.0 if relevant else 1.0)

        # --- generation metric ---
        ans = answer(q)["answer"]
        ok = _judge_answer(q, ans, expected)
        correct += int(ok)

        flag = "✓" if ok else "✗"
        print(f"{flag}  {q[:48]:<50}  retrieved@K={in_k}  reranked@N={in_n}")

    n = len(rows)
    print("\n── results ──")
    print(f"Retrieval hit@K : {hit_k}/{n}  ({hit_k / n:.0%})")
    print(f"Retrieval hit@N : {hit_n}/{n}  ({hit_n / n:.0%})   (after rerank)")
    print(f"MRR             : {mrr / n:.3f}")
    print(f"Answer correct  : {correct}/{n}  ({correct / n:.0%})   (LLM judge)")


if __name__ == "__main__":
    run()
