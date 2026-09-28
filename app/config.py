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
CHROMA_DIR = ROOT / ".chroma"  # on-disk vector store
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
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "claude-sonnet-5")
RERANK_MODEL = os.getenv("RERANK_MODEL", "rerank-multilingual-v3.0")

# --- Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
