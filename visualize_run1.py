import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import numpy as np

df = pd.read_csv("results/results.csv")
df1 = df[df['run'] == 1].copy()

strategy_order = ['FullMemory', 'SummaryMemory', 'GraphMemory', 'RAGMemory']
strategy_labels = ['Full Context', 'Summary', 'Knowledge Graph', 'RAG (Vector)']
colors = ['#2c3e50', '#2980b9', '#27ae60', '#e67e22']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Memory Compression Strategies — Trade-off Analysis (Run 1, llama-3.3-70b)', fontsize=15, fontweight='bold', y=0.98)

# --- Accuracy Bar Chart ---
ax = axes[0, 0]
accs = [df1[df1['strategy'] == s]['accuracy'].mean() for s in strategy_order]
bars = ax.bar(strategy_labels, accs, color=colors, edgecolor='white', linewidth=1.2)
ax.set_title('Mean Accuracy', fontsize=12, fontweight='bold')
ax.set_ylim(0, 1.05)
ax.set_ylabel('Accuracy (0–1)')
for bar, val in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# --- Latency Bar Chart ---
ax = axes[0, 1]
lats = [df1[df1['strategy'] == s]['latency_ms'].mean() for s in strategy_order]
bars = ax.bar(strategy_labels, lats, color=colors, edgecolor='white', linewidth=1.2)
ax.set_title('Mean Latency per Query', fontsize=12, fontweight='bold')
ax.set_ylabel('Latency (ms)')
for bar, val in zip(bars, lats):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, f'{val:.0f}ms', ha='center', va='bottom', fontsize=10, fontweight='bold')

# --- Cost Bar Chart ---
ax = axes[1, 0]
costs = [df1[df1['strategy'] == s]['cost_usd'].sum() for s in strategy_order]
bars = ax.bar(strategy_labels, [c * 1000 for c in costs], color=colors, edgecolor='white', linewidth=1.2)
ax.set_title('Total Cost per 20-Question Run', fontsize=12, fontweight='bold')
ax.set_ylabel('Cost (mUSD, x10⁻³)')
for bar, val in zip(bars, costs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1000 + 0.2, f'${val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# --- Heatmap ---
ax = axes[1, 1]
heatmap_data = pd.DataFrame({
    'Strategy': strategy_labels,
    'Accuracy': accs,
    'Latency': lats,
    'Cost': costs
}).set_index('Strategy')

# Normalize: higher is better for Accuracy; lower is better for Latency and Cost
heatmap_norm = heatmap_data.copy()
for col in heatmap_norm.columns:
    min_val = heatmap_norm[col].min()
    max_val = heatmap_norm[col].max()
    if max_val > min_val:
        if col == 'Accuracy':
            heatmap_norm[col] = (heatmap_norm[col] - min_val) / (max_val - min_val)
        else:
            heatmap_norm[col] = (max_val - heatmap_norm[col]) / (max_val - min_val)
    else:
        heatmap_norm[col] = 1.0

sns.heatmap(heatmap_norm, annot=True, cmap="YlGnBu", ax=ax, vmin=0, vmax=1, fmt='.2f', linewidths=0.5, linecolor='white')
ax.set_title('Normalized Performance (1.0 = Best)', fontsize=12, fontweight='bold')

plt.tight_layout(rect=[0, 0.02, 1, 0.95])
plt.savefig('results/dashboard_run1.png', dpi=200, bbox_inches='tight')
print("Saved results/dashboard_run1.png")

# --- Per-question accuracy heatmap ---
fig2, ax2 = plt.subplots(figsize=(12, 8))
pivot = df1.pivot(index='question_id', columns='strategy', values='accuracy')
pivot = pivot[strategy_order]
pivot.columns = strategy_labels
# Sort by question number
pivot = pivot.reindex([f'q{i}' for i in range(1, 21)])

sns.heatmap(pivot, annot=True, cmap="RdYlGn", vmin=0, vmax=1, fmt='.1f', linewidths=0.5, linecolor='white', ax=ax2)
ax2.set_title('Per-Question Accuracy by Strategy (Run 1, llama-3.3-70b)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Question ID')
ax2.set_xlabel('Memory Strategy')
plt.tight_layout()
plt.savefig('results/accuracy_heatmap.png', dpi=200, bbox_inches='tight')
print("Saved results/accuracy_heatmap.png")
