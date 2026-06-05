import json
import csv
import os
from tqdm import tqdm
from memory.full_memory import FullMemory
from memory.summary_memory import SummaryMemory
from memory.graph_memory import GraphMemory
from memory.rag_memory import RAGMemory
from agent.engine import AgentEngine
from eval.judge import evaluate_answer

def run_benchmark():
    os.makedirs("results", exist_ok=True)
    
    with open("data/conversation.json", "r") as f:
        conversation = json.load(f)
        
    with open("eval/questions.json", "r") as f:
        questions = json.load(f)

    strategies = {
        "FullMemory": FullMemory,
        "SummaryMemory": SummaryMemory,
        "GraphMemory": GraphMemory,
        "RAGMemory": RAGMemory
    }
    
    results = []
    
    # 1 run per strategy (deterministic eval with temp=0)
    for run in range(1, 2):
        for strat_name, MemoryClass in strategies.items():
            print(f"\n=== Running {strat_name} (Run {run}) ===")
            memory_instance = MemoryClass()
            
            # Populate memory with conversation
            print(f"Populating memory for {strat_name}...")
            for msg in tqdm(conversation, desc="Ingesting turns"):
                memory_instance.add(msg["role"], msg["content"])
                
            engine = AgentEngine(memory_instance)
            
            # Answer questions
            print("Evaluating questions...")
            for q in tqdm(questions, desc="Q&A"):
                q_id = q["question_id"]
                query_text = q["question"]
                ground_truth = q["ground_truth"]
                
                res = engine.answer_question(query_text)
                
                # Evaluate
                accuracy = evaluate_answer(query_text, ground_truth, res["answer"])
                
                results.append({
                    "strategy": strat_name,
                    "question_id": q_id,
                    "run": run,
                    "accuracy": accuracy,
                    "latency_ms": res["latency_ms"],
                    "prompt_tokens": res["prompt_tokens"],
                    "completion_tokens": res["completion_tokens"],
                    "cost_usd": res["cost_usd"]
                })

    with open("results/results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["strategy", "question_id", "run", "accuracy", "latency_ms", "prompt_tokens", "completion_tokens", "cost_usd"])
        writer.writeheader()
        for row in results:
            writer.writerow(row)
            
    print("\nBenchmark complete! Results saved to results/results.csv")

if __name__ == "__main__":
    run_benchmark()
