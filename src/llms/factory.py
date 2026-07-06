from config import LLM_PROVIDER

from llms.ollama_llm import OllamaLLM
from llms.groq_llm import GroqLLM


class LLMFactory:

    @staticmethod
    def create():

        provider = LLM_PROVIDER.lower()

        if provider == "ollama":
            return OllamaLLM()

        elif provider == "groq":
            return GroqLLM()

        raise ValueError(
            f"Unsupported LLM Provider: {LLM_PROVIDER}"
        )