import pandas as pd
import json
import os

df = pd.read_csv("results/results.csv")

# Filter only Run 1 (valid data — Runs 2 & 3 hit rate limits)
df1 = df[df['run'] == 1].copy()

print("=" * 70)
print("RUN 1 ANALYSIS (llama-3.3-70b-versatile)")
print("=" * 70)

# Load questions for context
with open("eval/questions.json") as f:
    questions = {q['question_id']: q for q in json.load(f)}

# --- ACCURACY ---
print("\n--- ACCURACY BY STRATEGY ---")
for strat in df1['strategy'].unique():
    sub = df1[df1['strategy'] == strat]
    mean_acc = sub['accuracy'].mean()
    perfect = (sub['accuracy'] == 1.0).sum()
    partial = (sub['accuracy'] == 0.5).sum()
    failed = (sub['accuracy'] == 0.0).sum()
    print(f"  {strat:20s}  Mean: {mean_acc:.3f}  |  Perfect: {perfect}  Partial: {partial}  Failed: {failed}")

# --- LATENCY ---
print("\n--- LATENCY BY STRATEGY (ms) ---")
for strat in df1['strategy'].unique():
    sub = df1[df1['strategy'] == strat]
    mean_lat = sub['latency_ms'].mean()
    min_lat = sub['latency_ms'].min()
    max_lat = sub['latency_ms'].max()
    print(f"  {strat:20s}  Mean: {mean_lat:>10.1f}ms  |  Min: {min_lat:>8.1f}ms  Max: {max_lat:>10.1f}ms")

# --- COST ---
print("\n--- COST BY STRATEGY (USD per 20-Q run) ---")
for strat in df1['strategy'].unique():
    sub = df1[df1['strategy'] == strat]
    total_cost = sub['cost_usd'].sum()
    total_prompt = sub['prompt_tokens'].sum()
    total_completion = sub['completion_tokens'].sum()
    print(f"  {strat:20s}  Total: ${total_cost:.6f}  |  Prompt Tokens: {total_prompt}  Completion: {total_completion}")

# --- PER-QUESTION BREAKDOWN ---
print("\n--- PER-QUESTION ACCURACY MATRIX ---")
print(f"  {'QID':>4}  {'Question':<55}  {'Full':>5}  {'Summ':>5}  {'Graph':>5}  {'RAG':>5}")
for qid in [f"q{i}" for i in range(1, 21)]:
    q_text = questions[qid]['question'][:52]
    scores = {}
    for strat in ['FullMemory', 'SummaryMemory', 'GraphMemory', 'RAGMemory']:
        row = df1[(df1['strategy'] == strat) & (df1['question_id'] == qid)]
        scores[strat] = row['accuracy'].values[0] if len(row) > 0 else '-'
    print(f"  {qid:>4}  {q_text:<55}  {scores['FullMemory']:>5}  {scores['SummaryMemory']:>5}  {scores['GraphMemory']:>5}  {scores['RAGMemory']:>5}")

# --- FAILURE ANALYSIS ---
print("\n--- FAILURE CASES (accuracy < 1.0) ---")
for _, row in df1[df1['accuracy'] < 1.0].iterrows():
    q = questions[row['question_id']]
    print(f"  [{row['strategy']}] {row['question_id']}: {q['question']}")
    print(f"    Ground Truth: {q['ground_truth']}")
    print(f"    Score: {row['accuracy']}")
    print()
