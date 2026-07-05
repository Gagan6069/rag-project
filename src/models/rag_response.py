from dataclasses import dataclass
from typing import List

from models.retrieval_result import RetrievalResult


@dataclass
class RAGResponse:

    question: str

    answer: str

    sources: List[RetrievalResult]