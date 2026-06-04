import time
from agent.llm import call_llm
from memory.base import BaseMemory

PRICE_PROMPT_1M = 0.59
PRICE_COMPLETION_1M = 0.79

def calculate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens / 1_000_000.0) * PRICE_PROMPT_1M + (completion_tokens / 1_000_000.0) * PRICE_COMPLETION_1M

class AgentEngine:
    def __init__(self, memory: BaseMemory):
        self.memory = memory

    def answer_question(self, query: str) -> dict:
        t0 = time.time()
        context = self.memory.retrieve(query)
        t_retrieve = time.time() - t0
        
        prompt = f"Context from Memory:\n{context}\n\nUser Question:\n{query}\n\nAnswer the user's question based strictly on the provided context."
        
        t1 = time.time()
        try:
            answer, pt, ct = call_llm(prompt, temperature=0.0)
        except Exception as e:
            answer = f"Error: {e}"
            pt = 0
            ct = 0
        t2 = time.time()
        
        latency = (t2 - t1) * 1000 # in ms
        
        query_cost = calculate_cost(pt, ct)
        
        return {
            "answer": answer,
            "latency_ms": latency,
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "cost_usd": query_cost
        }
