from langchain_community.vectorstores import FAISS

from src.embeddings import EmbeddingModel
from src.config import FAISS_INDEX_PATH, TOP_K

from src.models.retrieval_result import RetrievalResult
from src.retrievers.base_retriever import BaseRetriever


class FAISSRetriever(BaseRetriever):

    def __init__(self):

        if not FAISS_INDEX_PATH.exists():
            raise FileNotFoundError(
                "FAISS index was not found at "
                f"{FAISS_INDEX_PATH}. "
                "Run `python -m src.vectordb` first."
            )

        embedding_model = EmbeddingModel().get()

        self.db = FAISS.load_local(
            FAISS_INDEX_PATH,
            embedding_model,
            allow_dangerous_deserialization=True,
        )

    def retrieve(self, question: str):

        results = self.db.similarity_search_with_score(
            question,
            k=TOP_K
        )

        retrieval_results = []

        for doc, score in results:

            retrieval_results.append(
                RetrievalResult(
                    page_content=doc.page_content,
                    source=doc.metadata.get("source", "Unknown"),
                    page=doc.metadata.get("page", -1),
                    score=float(score),
                    chunk_id=doc.metadata.get("chunk_id", -1),
                    retrieval_method="faiss",
                )
            )

        return retrieval_results

    # Optional backward compatibility
    def search(self, question: str):
        return self.retrieve(question)