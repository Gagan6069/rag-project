from src.config import RETRIEVER_TYPE

from src.retrievers.faiss_retriever import FAISSRetriever
from src.retrievers.hybrid_retriever import HybridRetriever


class RetrieverFactory:

    @staticmethod
    def create():

        retriever_type = RETRIEVER_TYPE.lower()

        if retriever_type == "faiss":
            return FAISSRetriever()

        if retriever_type == "hybrid":
            return HybridRetriever()

        raise ValueError(
            f"Unsupported retriever type: {RETRIEVER_TYPE}"
        )