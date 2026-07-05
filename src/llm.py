from langchain_ollama import ChatOllama

from config import (
    OLLAMA_MODEL,
    TEMPERATURE,
)


class LocalLLM:

    def __init__(self):

        self.llm = ChatOllama(
            model=OLLAMA_MODEL,
            temperature=TEMPERATURE,
        )

    def ask(self, prompt):

        response = self.llm.invoke(prompt)

        return response.content