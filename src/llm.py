from langchain_ollama import ChatOllama
import requests

from config import (
    OLLAMA_MODEL,
    TEMPERATURE,
)

def check_ollama():
    try:
        requests.get("http://localhost:11434")
        return True
    except:
        return False

class LocalLLM:

    def __init__(self):

        self.llm = ChatOllama(
            model=OLLAMA_MODEL,
            temperature=TEMPERATURE,
        )

    def ask(self, prompt):
        
        if not check_ollama():
            raise Exception("Ollama is not running. Please run `ollama serve`")
        
        response = self.llm.invoke(prompt)

        return response.content
      