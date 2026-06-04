from abc import ABC, abstractmethod

class BaseMemory(ABC):
    @abstractmethod
    def add(self, role: str, content: str):
        """Add a new message to the memory."""
        pass

    @abstractmethod
    def retrieve(self, query: str) -> str:
        """Retrieve relevant context for the given query."""
        pass

    @abstractmethod
    def token_count(self) -> int:
        """Estimate the number of tokens currently stored/retrieved."""
        pass
