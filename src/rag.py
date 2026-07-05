from retrieve import Retriever
from llm import LocalLLM
from prompts import RAG_PROMPT
from context_builder import ContextBuilder
from builders.prompt_builder import PromptBuilder


class RAG:

    def __init__(self):

        self.retriever = Retriever()
        self.llm = LocalLLM()

    def ask(self, question):

        results = self.retriever.search(question)

        context = ContextBuilder.build(results)

        prompt = PromptBuilder.build(
            context,
            question
        )

        answer = self.llm.ask(prompt)

        return {
            "question": question,
            "answer": answer,
            "sources": results,
        }


if __name__ == "__main__":

    rag = RAG()

    while True:

        question = input("\nAsk: ")

        if question.lower() == "exit":
            break

        response = rag.ask(question)

        print("\n" + "=" * 60)
        print("QUESTION")
        print("=" * 60)
        print(response["question"])

        print("\n" + "=" * 60)
        print("ANSWER")
        print("=" * 60)
        print(response["answer"])

        print("\n" + "=" * 60)
        print("SOURCES")
        print("=" * 60)

        for doc, score in response["sources"]:

            print(f"Page      : {doc.metadata.get('page')}")
            print(f"Source    : {doc.metadata.get('source')}")
            print(f"Score     : {score:.4f}")
            print("-" * 40)