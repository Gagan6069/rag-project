from copy import deepcopy

from src.config import HYBRID_TOP_K, RRF_K

from src.retrievers.base_retriever import BaseRetriever
from src.retrievers.faiss_retriever import FAISSRetriever
from src.retrievers.bm25_retriever import BM25Retriever


class HybridRetriever(BaseRetriever):

    def __init__(self):

        self.faiss_retriever = FAISSRetriever()
        self.bm25_retriever = BM25Retriever()

    def retrieve(self, question: str):

        faiss_results = self.faiss_retriever.retrieve(question)

        bm25_results = self.bm25_retriever.retrieve(question)

        fused_results = self._reciprocal_rank_fusion(
            faiss_results,
            bm25_results
        )

        return fused_results[:HYBRID_TOP_K]

    def _reciprocal_rank_fusion(self, faiss_results, bm25_results):

        merged = {}

        self._add_results(
            merged=merged,
            results=faiss_results,
            method="faiss"
        )

        self._add_results(
            merged=merged,
            results=bm25_results,
            method="bm25"
        )

        fused = list(merged.values())

        fused.sort(
            key=lambda result: result.rrf_score,
            reverse=True
        )

        return fused

    def _add_results(self, merged, results, method):

        for rank, result in enumerate(results, start=1):

            key = self._get_key(result)

            rrf_score = 1 / (RRF_K + rank)

            if key not in merged:

                merged[key] = deepcopy(result)

                merged[key].rrf_score = 0.0

                merged[key].retrieval_method = "hybrid"

            existing = merged[key]

            existing.rrf_score += rrf_score

            if method == "faiss":
                existing.score = result.score

            if method == "bm25":
                existing.bm25_score = result.bm25_score

    def _get_key(self, result):

        if result.chunk_id != -1:
            return result.chunk_id

        return hash(result.page_content)