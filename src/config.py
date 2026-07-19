import os
from pathlib import Path

from dotenv import load_dotenv


# rag-project/src/config.py
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

load_dotenv(PROJECT_ROOT / ".env")


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


# ---------------------------------------------------------
# Embeddings
# ---------------------------------------------------------

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)


# ---------------------------------------------------------
# LLM
# ---------------------------------------------------------

LLM_PROVIDER = os.getenv(
    "LLM_PROVIDER",
    "groq",
).lower()

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b",
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.2:3b",
)

TEMPERATURE = float(
    os.getenv("TEMPERATURE", "0")
)


# ---------------------------------------------------------
# Chunking
# ---------------------------------------------------------

CHUNK_SIZE = int(
    os.getenv("CHUNK_SIZE", "500")
)

CHUNK_OVERLAP = int(
    os.getenv("CHUNK_OVERLAP", "100")
)


# ---------------------------------------------------------
# Retrieval
# ---------------------------------------------------------

RETRIEVER_TYPE = os.getenv(
    "RETRIEVER_TYPE",
    "hybrid",
).lower()

TOP_K = int(
    os.getenv("TOP_K", "10")
)

BM25_TOP_K = int(
    os.getenv("BM25_TOP_K", "10")
)

HYBRID_TOP_K = int(
    os.getenv("HYBRID_TOP_K", "10")
)

RRF_K = int(
    os.getenv("RRF_K", "60")
)


# ---------------------------------------------------------
# Reranking
# ---------------------------------------------------------

ENABLE_RERANKER = env_bool(
    "ENABLE_RERANKER",
    True,
)

RERANK_TOP_K = int(
    os.getenv("RERANK_TOP_K", "3")
)

RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)


# ---------------------------------------------------------
# Filtering
# ---------------------------------------------------------

_threshold = os.getenv("SIMILARITY_THRESHOLD")

SIMILARITY_THRESHOLD = (
    float(_threshold)
    if _threshold not in {None, "", "none", "null"}
    else None
)


# ---------------------------------------------------------
# Application paths
# ---------------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
STORAGE_DIR = PROJECT_ROOT / "storage"

PDF_PATH = DATA_DIR / "employee_handbook.pdf"

FAISS_INDEX_PATH = STORAGE_DIR / "faiss_index"

CHUNK_STORE_PATH = STORAGE_DIR / "chunk_store" / "chunks.json"

BENCHMARK_FILE = (
    SRC_DIR
    / "evaluation"
    / "benchmark_questions.json"
)

EVALUATION_OUTPUT_PATH = (
    OUTPUT_DIR
    / "run_results_v2.csv"
)


# ---------------------------------------------------------
# Debugging
# ---------------------------------------------------------

DEBUG = env_bool("DEBUG", True)


# Create writable directories automatically.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
CHUNK_STORE_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)