from memory.full_memory import FullMemory
from memory.summary_memory import SummaryMemory
from memory.graph_memory import GraphMemory
from memory.rag_memory import RAGMemory

def test_memories():
    mems = [FullMemory(), SummaryMemory(summary_every=2, keep_recent=1), GraphMemory(), RAGMemory()]
    
    for m in mems:
        print(f"=== Testing {m.__class__.__name__} ===")
        m.add("user", "My name is Alex")
        m.add("assistant", "Hi Alex!")
        m.add("user", "My favorite color is blue.")
        
        ret = m.retrieve("What is my name?")
        print(f"Retrieved Context:\n{ret}\n")

if __name__ == "__main__":
    test_memories()
