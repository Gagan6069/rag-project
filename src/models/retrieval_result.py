from dataclasses import dataclass


@dataclass
class RetrievalResult:

    page_content: str

    source: str

    page: int

    score: float