from langchain_ollama import ChatOllama

from src.llms.base import BaseLLM
from src.config import OLLAMA_MODEL


class OllamaLLM(BaseLLM):

    def __init__(self):

        self.llm = ChatOllama(
            model=OLLAMA_MODEL,
            temperature=0
        )

    def ask(self, prompt: str) -> str:

        return self.llm.invoke(prompt).content