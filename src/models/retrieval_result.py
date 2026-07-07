from dataclasses import dataclass


@dataclass
class RetrievalResult:

    page_content: str

    source: str

    page: int

    similarity_score: float
    
    rerank_score: float = 0.0