from builders.context_builder import ContextBuilder
from builders.prompt_builder import PromptBuilder

from retrievers.faiss_retriever import FAISSRetriever
from filters.score_filter import ScoreFilter

from models.rag_response import RAGResponse

from llm import LocalLLM

from config import SIMILARITY_THRESHOLD
from llms.groq_llm import GroqLLM
from llms.ollama_llm import OllamaLLM
from config import LLM_PROVIDER

from utils.logger import logger
from debug.rag_dashboard import RAGDashboard

from llms.factory import LLMFactory

class RAGService:

    def __init__(self):

        self.retriever = FAISSRetriever()
        self.context_builder = ContextBuilder()
        # self.llm = LocalLLM()
        # if LLM_PROVIDER == "groq":
        #     self.llm = GroqLLM()

        # elif LLM_PROVIDER == "ollama":
        #     self.llm = OllamaLLM()

        # else:
        #     raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")
        self.llm = LLMFactory.create()

    def ask(self, question):

        logger.info(f"Received question: {question}")

        # -----------------------------
        # Retrieve Documents
        # -----------------------------
        logger.info("Retrieving documents...")

        results = self.retriever.search(question)

        logger.info(f"Retrieved {len(results)} chunks")

        RAGDashboard.retrieval(results)

        # -----------------------------
        # Score Filtering
        # -----------------------------
        results = ScoreFilter.filter(
            results,
            SIMILARITY_THRESHOLD
        )

        logger.info(f"{len(results)} chunks remained after filtering")

        if not results:

            logger.warning("No relevant documents found.")

            return RAGResponse(
                question=question,
                answer="I couldn't find relevant information in the document.",
                sources=[]
            )

        # -----------------------------
        # Build Context
        # -----------------------------
        logger.info("Building context...")

        context = self.context_builder.build(results)

        RAGDashboard.context(context)

        # -----------------------------
        # Build Prompt
        # -----------------------------
        logger.info("Building prompt...")

        prompt = PromptBuilder.build(
            context=context,
            question=question
        )

        RAGDashboard.prompt(prompt)

        # -----------------------------
        # Call LLM
        # -----------------------------
        logger.info("Calling LLM...")

        answer = self.llm.ask(prompt)

        RAGDashboard.answer(answer)

        logger.info("Answer generated successfully.")

        # -----------------------------
        # Return Response
        # -----------------------------
        return RAGResponse(
            question=question,
            answer=answer,
            sources=results
        )