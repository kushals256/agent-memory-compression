from .base import BaseMemory
from agent.llm import call_llm

class SummaryMemory(BaseMemory):
    def __init__(self, summary_every=10, keep_recent=5):
        self.messages = []
        self.summary = "No summary yet."
        self.summary_every = summary_every
        self.keep_recent = keep_recent
        self.turns = 0
        
        self.memory_prompt_tokens = 0
        self.memory_completion_tokens = 0

    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        self.turns += 1
        
        if self.turns % self.summary_every == 0:
            self._update_summary()

    def _update_summary(self):
        # Summarize everything except the keep_recent
        if len(self.messages) <= self.keep_recent:
            return
            
        messages_to_summarize = self.messages[:-self.keep_recent]
        recent_messages = self.messages[-self.keep_recent:]
        
        context_str = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages_to_summarize])
        
        prompt = f"Current Summary:\n{self.summary}\n\nNew Messages to incorporate into the summary:\n{context_str}\n\nWrite an updated, concise summary of the key facts from this conversation."
        
        try:
            new_summary, pt, ct = call_llm(prompt, system_prompt="You are an expert summarizer. Extract and retain all specific facts, preferences, and personal details.")
            self.summary = new_summary
            self.memory_prompt_tokens += pt
            self.memory_completion_tokens += ct
        except Exception as e:
            print(f"Error updating summary: {e}")
            
        # Update our stored messages to only include the recent ones
        self.messages = recent_messages

    def retrieve(self, query: str) -> str:
        context = [f"[System Summary of Past Conversation]\n{self.summary}\n\n[Recent Conversation]"]
        for msg in self.messages:
            context.append(f"{msg['role'].capitalize()}: {msg['content']}")
        return "\n".join(context)

    def token_count(self) -> int:
        text = self.retrieve("")
        return int(len(text.split()) * 1.3)
