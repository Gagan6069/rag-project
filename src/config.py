# Embedding Model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Ollama
# OLLAMA_MODEL = "gemma3:4b"
OLLAMA_MODEL = "llama3.2:3b"
TEMPERATURE = 0

# Chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Retrieval
TOP_K = 3
SIMILARITY_THRESHOLD = 0.80

# Paths
FAISS_INDEX_PATH = "faiss_index"
PDF_PATH = "../data/employee_handbook.pdf"