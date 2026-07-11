from dataclasses import dataclass


@dataclass
class RetrievalResult:

    page_content: str

    source: str

    page: int

    score: float = 0.0              # FAISS similarity/distance score

    chunk_id: int = -1

    bm25_score: float = 0.0

    rrf_score: float = 0.0

    rerank_score: float = 0.0

    retrieval_method: str = "faiss"