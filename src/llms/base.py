from abc import ABC, abstractmethod


class BaseLLM(ABC):

    @abstractmethod
    def ask(self, prompt: str) -> str:
        pass