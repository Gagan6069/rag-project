from abc import ABC, abstractmethod


class BaseRetriever(ABC):

    @abstractmethod
    def retrieve(self, question: str):
        """
        Returns a list of RetrievalResult objects.
        """
        pass