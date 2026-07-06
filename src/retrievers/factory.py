from config import RETRIEVER_TYPE

from retrievers.faiss_retriever import FAISSRetriever


class RetrieverFactory:

    @staticmethod
    def create():

        retriever = RETRIEVER_TYPE.lower()

        if retriever == "faiss":
            return FAISSRetriever()

        raise ValueError(
            f"Unsupported retriever: {RETRIEVER_TYPE}"
        )