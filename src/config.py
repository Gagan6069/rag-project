# Embedding Model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

##########################################
#LLM selection
##########################################
LLM_PROVIDER = "groq"   # or "ollama"

# Ollama
# OLLAMA_MODEL = "gemma3:4b"
OLLAMA_MODEL = "llama3.2:3b"
TEMPERATURE = 0

#Groq
GROQ_MODEL = "openai/gpt-oss-20b"

##########################################
# Retriever
##########################################

RETRIEVER_TYPE = "faiss"

# Chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Retrieval
TOP_K = 10
SIMILARITY_THRESHOLD = 0.80

##################################
# Reranking
##################################

ENABLE_RERANKER = True

RERANK_TOP_K = 3


# Paths
FAISS_INDEX_PATH = "faiss_index"
PDF_PATH = "../data/employee_handbook.pdf"

DEBUG = False
