from .base import BaseMemory

class FullMemory(BaseMemory):
    def __init__(self):
        self.messages = []

    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

    def retrieve(self, query: str) -> str:
        # For full memory, we return the entire history as a formatted string
        context = []
        for msg in self.messages:
            context.append(f"{msg['role'].capitalize()}: {msg['content']}")
        return "\n".join(context)

    def token_count(self) -> int:
        # Rough estimation: ~1.3 tokens per word
        text = self.retrieve("")
        return int(len(text.split()) * 1.3)
