# זכותון — Hebrew Rights RAG Assistant

A retrieval-augmented (RAG) assistant that answers questions in **Hebrew** about citizens'
rights and benefits, grounded in a knowledge base (modeled on כל-זכות / Kol-Zchut). Every
answer is built **only** from retrieved sources, with **citations**, and says *"לא נמצאה
תשובה"* when the knowledge base doesn't cover the question — no hallucinated legal advice.

This is a portfolio POC that exercises the full enterprise-RAG stack end to end:
**chunking → metadata → embeddings → vector DB → retrieval → reranking → grounded
generation → evaluation → API → UI.** It is intentionally in **Python (FastAPI)**.

> Why Hebrew? Most RAG demos are English. Hebrew RAG is genuinely harder (embeddings,
> chunking, and reranking all degrade on Hebrew), so doing it well is a real signal — and
> it's exactly what an Israeli enterprise assistant needs.

---

## Architecture (this is also the interview cheat-sheet)

```
                       ┌─────────────── INGESTION (offline, run once) ───────────────┐
   docs (Hebrew) ─▶ chunking ─▶ metadata ─▶ embeddings ─▶ vector store (Qdrant)
                       └──────────────────────────────────────────────────────────────┘

                       ┌─────────────────── QUERY (per question) ───────────────────┐
   follow-up ─▶ CONDENSE w/ history ─▶ standalone query ─▶ embed ─▶ retrieve top-K
                                                                      │
                                                                      ▼
   history + summary ──────────────────────────▶ RERANK top-N ─▶ build context
                                                                      │
                                                                      ▼
                                             Claude (grounded, cited, "I don't know")
                                                                      │
                                                                      ▼
                                        store turn ─▶ memory (window + rolling summary)
                       └──────────────────────────────────────────────────────────────┘
```

### The pipeline, stage by stage

| Stage | File | What it does | Why it matters |
|---|---|---|---|
| **Conversation memory** | `app/memory.py` | Per-session history: last N turns verbatim + a rolling summary of older ones | Multi-turn context without an unbounded prompt. Enables follow-ups ("וכמה מגיע לי?") |
| **Query condensing** | `app/rag.py` | Rewrites a follow-up + history → one standalone query **before** retrieval (fast model) | A follow-up alone retrieves nothing; you must resolve coreferences first, or retrieval fails |
| **Chunking** | `app/chunking.py` | Splits each doc into ~overlapping chunks | Too big → noisy retrieval; too small → lost context. Overlap keeps context across boundaries. |
| **Metadata** | `app/chunking.py` | Attaches `title, category, source_url, doc_id, chunk_index` to every chunk | Enables **filtering** (by category/permissions) and **citations** |
| **Embeddings** | `app/embeddings.py` | Turns text → vector (Cohere `embed-multilingual-v3.0`) | Multilingual (strong on Hebrew); asymmetric `search_document`/`search_query` types for docs vs query |
| **Vector DB** | `app/vectorstore.py` | Stores vectors + metadata, does ANN search (Qdrant, local mode) | The searchable index; metadata travels with each vector. Embedded — no server, no C++ build |
| **Retrieval** | `app/retrieval.py` | Embeds the query, returns top-K similar chunks | Recall-oriented: cast a wide net |
| **Reranking** | `app/rerank.py` | Re-scores the top-K with a cross-encoder (Cohere Rerank) → top-N | Precision: dense search is fuzzy; a reranker reads query+chunk **together** and sorts by true relevance |
| **Generation** | `app/rag.py` | Stuffs the top-N into a Hebrew prompt; Claude answers grounded + citations | Grounding + "לא נמצאה תשובה" prevents hallucination |
| **Evaluation** | `app/eval.py` | Runs a golden Q&A set; measures retrieval (recall@k, MRR) + answer faithfulness (LLM-judge) | *"How do you know quality improved?"* — measure before/after every change |
| **API** | `app/api.py` | `POST /ask` (answer) and `POST /search` (retrieval debug) | Backend service other systems call |
| **UI** | `ui/index.html` | Ask a question; see the answer **and** the retrieved chunks + scores | Eval/debug console for root-cause analysis |

### Key design decisions (say these in the interview)
- **Grounding over recall of the model's memory:** the answer uses only retrieved context; the
  prompt forbids outside knowledge and requires *"לא נמצאה תשובה"* when unsupported. This is the
  #1 guard against hallucinated legal advice.
- **Retrieval is recall, reranking is precision:** dense vector search returns a wide top-K (fast,
  fuzzy); the cross-encoder reranker then reads each (query, chunk) pair and re-sorts, so the 3-4
  chunks that reach the model are the truly relevant ones. This is the single biggest quality lever.
- **Metadata is first-class:** every chunk carries its source, so answers cite, and retrieval can
  filter (e.g. per-tenant / per-permission in a real enterprise system).
- **Conversational RAG, not stateless Q&A:** the retrieval query and the user's utterance are *not*
  the same thing in a multi-turn chat. "וכמה מגיע לי?" only makes sense given prior turns, so we
  **condense** (history + follow-up) into a standalone query before retrieving. Memory is bounded —
  last N turns verbatim plus a rolling summary — so cost/latency stay flat over a long chat. The
  cheap sub-tasks (condense, summarize) run on a small fast model; only the grounded answer uses the
  strong one.
- **Evaluate retrieval and generation separately:** a wrong answer is either a *retrieval* failure
  (the right chunk wasn't found) or a *generation* failure (it was found but the model ignored/
  misread it). The `/search` view + the eval metrics tell you which — that's root-cause analysis.

---

## Run it

### 1. Install
```bash
cd zchut-rag
python -m venv .venv
# activate the venv:
#   Windows PowerShell : .venv\Scripts\Activate.ps1
#   Windows cmd        : .venv\Scripts\activate.bat
#   macOS / Linux      : source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Keys
```bash
cp .env.example .env
# then edit .env and set:
#   COHERE_API_KEY    (embeddings + reranking) — required; free trial key at dashboard.cohere.com
#   ANTHROPIC_API_KEY (answer generation)      — required
```
A single Cohere key drives both embeddings and reranking; Claude writes the answer.
(Cohere trial keys are capped at ~10 requests/min — `app/retry.py` handles the 429s with
exponential backoff, so the eval still completes, just more slowly.)

### 3. Ingest the corpus (chunk → embed → store)
```bash
python -m app.ingest
```

### 4. Serve the API + UI
```bash
uvicorn app.api:app --port 8000
# open http://localhost:8000  (UI)   ·   POST http://localhost:8000/ask  (API)
```
> Qdrant's local mode holds an exclusive lock on `.qdrant/`, so run one process at a time:
> stop the server before re-running `python -m app.ingest`. (That's why we skip `--reload`.)

### 5. Evaluate
```bash
python -m app.eval
# prints retrieval metrics (recall@k, MRR) and answer faithfulness on the golden set
```

---

## Swap points (production notes)
- **Embeddings:** swap Cohere for a local multilingual model (`intfloat/multilingual-e5-large`
  or `BAAI/bge-m3` via `sentence-transformers`) — no API key, no rate limits, better data control.
- **Vector DB:** the code already uses Qdrant; swap local mode for a **Qdrant server** (Docker)
  or **pgvector** (reuse a Postgres you already run) for scale + metadata filtering at query time.
  Only `app/vectorstore.py` changes — point `QdrantClient` at a URL instead of a path.
- **Retrieval:** add **hybrid** search (dense + BM25 keyword) — helps recall on names/numbers that
  embeddings miss.
- **Reranking:** swap Cohere for a local `BAAI/bge-reranker-v2-m3` cross-encoder.
