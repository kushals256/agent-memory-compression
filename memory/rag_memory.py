import chromadb
from chromadb.utils import embedding_functions
from .base import BaseMemory

class RAGMemory(BaseMemory):
    def __init__(self):
        self.chroma_client = chromadb.Client()
        self.collection_name = "agent_memory"
        
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
        except:
            pass
            
        self.collection = self.chroma_client.create_collection(name=self.collection_name, embedding_function=self.ef)
        
        self.msg_count = 0

    def add(self, role: str, content: str):
        doc = f"{role.capitalize()}: {content}"
        self.collection.add(
            documents=[doc],
            metadatas=[{"role": role, "turn": self.msg_count}],
            ids=[f"msg_{self.msg_count}"]
        )
        self.msg_count += 1

    def retrieve(self, query: str) -> str:
        if self.msg_count == 0:
            return ""
            
        k = min(5, self.msg_count)
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )
        
        documents = results['documents'][0]
        return "Relevant past messages:\n" + "\n".join(documents)

    def token_count(self) -> int:
        text = self.retrieve("")
        return int(len(text.split()) * 1.3)
