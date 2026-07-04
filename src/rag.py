from retrieve import Retriever
from llm import LocalLLM


class RAG:

    def __init__(self):

        self.retriever = Retriever()
        self.llm = LocalLLM()

    def ask(self, question):

        docs = self.retriever.search(question)

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        prompt = f"""
You are a helpful assistant.

Answer ONLY from the provided context.

If the answer is not present,
say "I don't know."

Context:
{context}

Question:
{question}

Answer:
"""

        return self.llm.ask(prompt)


if __name__ == "__main__":

    rag = RAG()

    while True:

        question = input("\nAsk: ")

        if question.lower() == "exit":
            break

        print("\nAnswer:\n")

        print(rag.ask(question))