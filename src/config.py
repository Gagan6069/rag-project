# Embedding Model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# LLM
LLM_PROVIDER = "groq"      # "groq" or "ollama"
OLLAMA_MODEL = "llama3.2:3b"
GROQ_MODEL = "openai/gpt-oss-20b"
TEMPERATURE = 0

# Chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Retrieval
RETRIEVER_TYPE = "hybrid"  # "faiss" or "hybrid"

TOP_K = 10                # FAISS candidate count
BM25_TOP_K = 10           # BM25 candidate count
HYBRID_TOP_K = 10         # candidates after merging

SIMILARITY_THRESHOLD = None

# RRF
RRF_K = 60

# Reranking
ENABLE_RERANKER = True
RERANK_TOP_K = 3

# Paths
FAISS_INDEX_PATH = "faiss_index"
PDF_PATH = "../data/employee_handbook.pdf"
CHUNK_STORE_PATH = "chunk_store/chunks.json"

DEBUG = True