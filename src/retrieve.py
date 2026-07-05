from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    TOP_K,
)


class Retriever:

    def __init__(self):

        self.embedding = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

        self.db = FAISS.load_local(
            FAISS_INDEX_PATH,
            self.embedding,
            allow_dangerous_deserialization=True,
        )

    def search(self, question):

        results = self.db.similarity_search_with_score(
            question,
            k=TOP_K,
        )

        return results


if __name__ == "__main__":

    retriever = Retriever()

    results = retriever.search(
        "How many casual leaves are allowed?"
    )

    for doc, score in results:

        print("=" * 50)
        print(f"Similarity Score : {score}")
        print(f"Page             : {doc.metadata.get('page')}")
        print(f"Source           : {doc.metadata.get('source')}")
        print(doc.page_content[:250])