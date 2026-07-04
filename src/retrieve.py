from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


class Retriever:

    def __init__(self):

        self.embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.db = FAISS.load_local(
            "faiss_index",
            self.embedding,
            allow_dangerous_deserialization=True
        )

    def search(self, question, k=3):

        docs = self.db.similarity_search(
            question,
            k=k
        )

        return docs


if __name__ == "__main__":

    retriever = Retriever()

    docs = retriever.search(
        "How many paid leaves are allowed?"
    )

    for i, doc in enumerate(docs, start=1):
        print(f"\n----- Document {i} -----\n")
        print(doc.page_content)