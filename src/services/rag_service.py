from retrieve import Retriever
from llm import LocalLLM
from prompts import RAG_PROMPT
from context_builder import ContextBuilder
from builders.prompt_builder import PromptBuilder
from config import SIMILARITY_THRESHOLD
from filters.score_filter import ScoreFilter

class RAGService:

    def ask(self, question):

        results = self.retriever.search(question)

        results = ScoreFilter.filter(
            results,
            SIMILARITY_THRESHOLD
        )

        # Business decision
        if not results:
            return {
                "question": question,
                "answer": "I couldn't find relevant information in the document.",
                "sources": []
            }

        context = ContextBuilder.build(results)

        prompt = PromptBuilder.build(
            context,
            question
        )

        answer = self.llm.ask(prompt)

        return {
            "question": question,
            "answer": answer,
            "sources": results
        }