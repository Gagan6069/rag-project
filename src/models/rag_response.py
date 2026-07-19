from dataclasses import dataclass
from typing import List

from src.models.retrieval_result import RetrievalResult


@dataclass
class RAGResponse:

    question: str

    answer: str

    sources: List[RetrievalResult]

    context: str = ""

    prompt: str = ""