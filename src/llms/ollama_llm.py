from langchain_ollama import ChatOllama

from llms.base import BaseLLM
from config import OLLAMA_MODEL


class OllamaLLM(BaseLLM):

    def __init__(self):

        self.llm = ChatOllama(
            model=OLLAMA_MODEL,
            temperature=0
        )

    def ask(self, prompt: str) -> str:

        return self.llm.invoke(prompt).content