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
   docs (Hebrew) ─▶ chunking ─▶ metadata ─▶ embeddings ─▶ vector store (Chroma)
                       └──────────────────────────────────────────────────────────────┘

                       ┌─────────────────── QUERY (per question) ───────────────────┐
   question ─▶ embed ─▶ dense retrieve top-K ─▶ RERANK top-N ─▶ build context
                                                                      │
                                                                      ▼
                                             Claude (grounded, cited, "I don't know")
                       └──────────────────────────────────────────────────────────────┘
```

### The pipeline, stage by stage

| Stage | File | What it does | Why it matters |
|---|---|---|---|
| **Chunking** | `app/chunking.py` | Splits each doc into ~overlapping chunks | Too big → noisy retrieval; too small → lost context. Overlap keeps context across boundaries. |
| **Metadata** | `app/chunking.py` | Attaches `title, category, source_url, doc_id, chunk_index` to every chunk | Enables **filtering** (by category/permissions) and **citations** |
| **Embeddings** | `app/embeddings.py` | Turns text → vector (OpenAI `text-embedding-3-small`) | Same model for docs **and** query; multilingual so Hebrew works |
| **Vector DB** | `app/vectorstore.py` | Stores vectors + metadata, does ANN search (Chroma) | The searchable index; metadata travels with each vector |
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
- **Evaluate retrieval and generation separately:** a wrong answer is either a *retrieval* failure
  (the right chunk wasn't found) or a *generation* failure (it was found but the model ignored/
  misread it). The `/search` view + the eval metrics tell you which — that's root-cause analysis.

---

## Run it

### 1. Install
```bash
cd zchut-rag
python -m venv .venv && source .venv/Scripts/activate   # Windows Git Bash; use .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```

### 2. Keys
```bash
cp .env.example .env
# then edit .env and set:
#   OPENAI_API_KEY   (embeddings)          — required
#   ANTHROPIC_API_KEY (answer generation)  — required
#   COHERE_API_KEY   (reranking)           — optional; without it, reranking falls back to score order
```

### 3. Ingest the corpus (chunk → embed → store)
```bash
python -m app.ingest
```

### 4. Serve the API + UI
```bash
uvicorn app.api:app --reload
# open http://localhost:8000  (UI)   ·   POST http://localhost:8000/ask  (API)
```

### 5. Evaluate
```bash
python -m app.eval
# prints retrieval metrics (recall@k, MRR) and answer faithfulness on the golden set
```

---

## Swap points (production notes)
- **Embeddings:** swap OpenAI for a local multilingual model (`intfloat/multilingual-e5-large`
  or `BAAI/bge-m3` via `sentence-transformers`) — no API key, better data control.
- **Vector DB:** swap Chroma for **pgvector** (reuse a Postgres you already run) or Qdrant/Pinecone
  for scale + metadata filtering at query time.
- **Retrieval:** add **hybrid** search (dense + BM25 keyword) — helps recall on names/numbers that
  embeddings miss.
- **Reranking:** swap Cohere for a local `BAAI/bge-reranker-v2-m3` cross-encoder.
