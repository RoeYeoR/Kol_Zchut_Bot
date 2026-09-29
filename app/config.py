"""Central configuration. All tunables live here so the pipeline stays declarative."""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Windows terminals default to cp1252 and crash when printing Hebrew. Force UTF-8
# stdout/stderr so the CLI scripts (ingest, rag, eval) print Hebrew cleanly.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "data" / "corpus"
EVAL_FILE = ROOT / "data" / "eval" / "golden.jsonl"
STORE_DIR = ROOT / ".qdrant"  # on-disk vector store (Qdrant local mode)
COLLECTION = "zchut"

# --- Chunking ---
# Sizes are in characters (simple + language-agnostic). Hebrew has no easy token
# splitter, so we chunk on paragraph/sentence boundaries with an overlap.
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120

# --- Retrieval / rerank ---
RETRIEVE_K = 12   # wide net from the vector DB (recall)
RERANK_N = 4      # what actually reaches the model after reranking (precision)

# --- Models ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embed-multilingual-v3.0")
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "claude-sonnet-5")
RERANK_MODEL = os.getenv("RERANK_MODEL", "rerank-multilingual-v3.0")
# Cheap/fast model for the auxiliary LLM steps (query condensing + history summarization).
# Routing simple sub-tasks to a small model keeps latency and cost down; the strong model
# is reserved for the actual grounded answer.
CONDENSE_MODEL = os.getenv("CONDENSE_MODEL", "claude-haiku-4-5")

# --- Keys ---
# Cohere powers BOTH embeddings and reranking; Claude writes the answer. No OpenAI needed.
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
