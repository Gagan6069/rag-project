import json
import re

from rank_bm25 import BM25Okapi

from src.config import CHUNK_STORE_PATH, BM25_TOP_K
from src.models.retrieval_result import RetrievalResult
from src.retrievers.base_retriever import BaseRetriever


class BM25Retriever(BaseRetriever):

    def __init__(self):

        self.chunks = self._load_chunks()

        self.tokenized_corpus = [
            self._tokenize(chunk["page_content"])
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _load_chunks(self):

        if not CHUNK_STORE_PATH.exists():
            raise FileNotFoundError(
                "BM25 chunk store was not found at "
                f"{CHUNK_STORE_PATH}. "
                "Run `python -m src.vectordb` first."
            )

        with open(CHUNK_STORE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    def _tokenize(self, text: str):

        return re.findall(
            r"\b\w+\b",
            text.lower()
        )

    def retrieve(self, question: str):

        query_tokens = self._tokenize(question)

        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        results = []

        for index in ranked_indices[:BM25_TOP_K]:

            score = float(scores[index])

            if score <= 0:
                continue

            chunk = self.chunks[index]
            metadata = chunk.get("metadata", {})

            results.append(
                RetrievalResult(
                    page_content=chunk.get("page_content", ""),
                    source=metadata.get("source", "Unknown"),
                    page=metadata.get("page", -1),
                    score=0.0,
                    chunk_id=metadata.get("chunk_id", -1),
                    bm25_score=score,
                    retrieval_method="bm25",
                )
            )

        return results