import json
from agent.llm import call_llm

def evaluate_answer(question: str, ground_truth: str, agent_answer: str) -> float:
    prompt = f"""
Evaluate the agent's answer against the ground truth.
Question: {question}
Ground Truth: {ground_truth}
Agent Answer: {agent_answer}

Rubric:
1.0: Fully correct, no hallucination
0.5: Partially correct or incomplete
0.0: Wrong, hallucinated, or "I don't know"

Output ONLY valid JSON in this format:
{{"score": 1.0, "reason": "Explain why"}}
"""
    try:
        response, _, _ = call_llm(prompt, system_prompt="You are an impartial judge scoring AI answers. Output only JSON.", temperature=0.0)
        
        start = response.find('{')
        end = response.rfind('}') + 1
        if start != -1 and end != -1:
            res_json = json.loads(response[start:end])
            return float(res_json.get("score", 0.0))
        return 0.0
    except Exception as e:
        # print(f"Eval Error: {e}")
        return 0.0
