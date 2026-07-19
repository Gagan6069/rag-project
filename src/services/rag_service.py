from src.builders.context_builder import ContextBuilder
from src.builders.prompt_builder import PromptBuilder

from src.retrievers.faiss_retriever import FAISSRetriever
from src.filters.score_filter import ScoreFilter

from src.models.rag_response import RAGResponse

from src.llm import LocalLLM

from src.config import SIMILARITY_THRESHOLD
from src.llms.groq_llm import GroqLLM
from src.llms.ollama_llm import OllamaLLM
from src.config import LLM_PROVIDER

from src.utils.logger import logger
from src.debug.rag_dashboard import RAGDashboard

from src.llms.factory import LLMFactory

from src.retrievers.factory import RetrieverFactory
from src.rerankers.cross_encoder_reranker import CrossEncoderReranker
from src.config import ENABLE_RERANKER
from src.config import RERANK_TOP_K


class RAGService:

    def __init__(self):

        # self.retriever = FAISSRetriever()
        self.retriever = RetrieverFactory.create()
        self.context_builder = ContextBuilder()
        # self.llm = LocalLLM()
        # if LLM_PROVIDER == "groq":
        #     self.llm = GroqLLM()

        # elif LLM_PROVIDER == "ollama":
        #     self.llm = OllamaLLM()

        # else:
        #     raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")
        self.llm = LLMFactory.create()
        
        if ENABLE_RERANKER:
            self.reranker = CrossEncoderReranker()

    def ask(self, question):

        logger.info(f"Received question: {question}")

        # -----------------------------
        # Retrieve Documents
        # -----------------------------
        logger.info("Retrieving documents...")

        results = self.retriever.retrieve(question)
        
        if ENABLE_RERANKER:

            results = self.reranker.rerank(
                question,
                results
            )


        results = results[:RERANK_TOP_K]

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
                answer=answer,
                sources=results,
                context=context,
                prompt=prompt
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