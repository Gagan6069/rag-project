import os

from src.config import (
    CHUNK_STORE_PATH,
    FAISS_INDEX_PATH,
    GROQ_MODEL,
    LLM_PROVIDER,
    PDF_PATH,
)


class StartupValidator:

    @staticmethod
    def validate_for_indexing() -> None:

        if not PDF_PATH.exists():
            raise FileNotFoundError(
                f"PDF not found: {PDF_PATH}"
            )

    @staticmethod
    def validate_for_chat() -> None:

        errors = []

        if not FAISS_INDEX_PATH.exists():
            errors.append(
                "FAISS index is missing. Run: "
                "python -m src.vectordb"
            )

        if not CHUNK_STORE_PATH.exists():
            errors.append(
                "BM25 chunk store is missing. Run: "
                "python -m src.vectordb"
            )

        if LLM_PROVIDER == "groq":

            if not os.getenv("GROQ_API_KEY"):
                errors.append(
                    "GROQ_API_KEY is missing from .env."
                )

            if not GROQ_MODEL:
                errors.append(
                    "GROQ_MODEL is not configured."
                )

        if errors:
            message = "\n- ".join(errors)

            raise RuntimeError(
                "Startup validation failed:\n- "
                + message
            )