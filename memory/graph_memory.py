import networkx as nx
import json
from .base import BaseMemory
from agent.llm import call_llm

class GraphMemory(BaseMemory):
    def __init__(self):
        self.graph = nx.DiGraph()
        self.memory_prompt_tokens = 0
        self.memory_completion_tokens = 0

    def add(self, role: str, content: str):
        prompt = f"""
Extract all factual relationships from this message as a JSON list of arrays. Each array should be [Subject, Predicate, Object].
If there are no facts, return an empty list [].
Message ({role}): "{content}"

Example format:
[
  ["User", "has name", "Alex"],
  ["User", "likes", "pizza"]
]
Return ONLY valid JSON array and nothing else.
"""
        try:
            response, pt, ct = call_llm(prompt, system_prompt="You are a strict JSON data extractor. Output only the requested JSON array and nothing else.", temperature=0.0)
            self.memory_prompt_tokens += pt
            self.memory_completion_tokens += ct
            
            # Clean up response to get just the JSON part
            start = response.find('[')
            end = response.rfind(']') + 1
            if start != -1 and end != -1:
                json_str = response[start:end]
                triples = json.loads(json_str)
                for triple in triples:
                    if len(triple) == 3:
                        subj, pred, obj = [str(x) for x in triple]
                        # For contradiction handling: if subject and predicate exist, remove the old edge
                        edges_to_remove = []
                        for u, v, d in self.graph.out_edges(subj, data=True):
                            if d.get('predicate') == pred:
                                edges_to_remove.append((u, v))
                        self.graph.remove_edges_from(edges_to_remove)
                        
                        # Add new edge
                        self.graph.add_edge(subj, obj, predicate=pred)
        except Exception as e:
            pass

    def retrieve(self, query: str) -> str:
        prompt = f"""
Extract the key entities from this query as a JSON list of strings.
Query: "{query}"

Also include direct synonyms or specific instances of the entities to aid in matching (e.g., if looking for 'siblings', also include 'brother' and 'sister').

Example format:
["User", "pizza", "brother", "sister"]
Return ONLY valid JSON.
"""
        try:
            response, pt, ct = call_llm(prompt, system_prompt="You are a strict JSON data extractor. Output only the requested JSON array.")
            self.memory_prompt_tokens += pt
            self.memory_completion_tokens += ct
            
            start = response.find('[')
            end = response.rfind(']') + 1
            if start != -1 and end != -1:
                json_str = response[start:end]
                entities = json.loads(json_str)
            else:
                entities = []
        except:
            entities = []
            
        extracted_triples = []
        for entity in entities:
            matched_nodes = [n for n in self.graph.nodes() if str(entity).lower() in str(n).lower() or str(n).lower() in str(entity).lower()]
            # If "user" is not explicitly in entity but is often the subject, add "User" or "Alex" implicitly if it's about them
            
            for node in matched_nodes:
                for u, v, d in self.graph.out_edges(node, data=True):
                    extracted_triples.append(f"({u}, {d.get('predicate', 'related_to')}, {v})")
                for u, v, d in self.graph.in_edges(node, data=True):
                    extracted_triples.append(f"({u}, {d.get('predicate', 'related_to')}, {v})")
                    
        # Remove duplicates
        extracted_triples = list(set(extracted_triples))
        
        if not extracted_triples:
            all_edges = list(self.graph.edges(data=True))
            extracted_triples = [f"({u}, {d.get('predicate', 'related_to')}, {v})" for u, v, d in all_edges[:50]]

        return "Knowledge Graph Facts:\n" + "\n".join(extracted_triples)

    def token_count(self) -> int:
        text = self.retrieve("")
        return int(len(text.split()) * 1.3)
