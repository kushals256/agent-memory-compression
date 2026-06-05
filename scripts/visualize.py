import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def visualize():
    if not os.path.exists("results/results.csv"):
        print("results/results.csv not found. Run benchmark.py first.")
        return

    df = pd.read_csv("results/results.csv")
    
    acc_stats = df.groupby('strategy')['accuracy'].agg(['mean', 'std']).reset_index()
    lat_stats = df.groupby('strategy')['latency_ms'].agg(['mean', 'std']).reset_index()
    
    cost_stats = df.groupby(['strategy', 'run'])['cost_usd'].sum().reset_index()
    cost_agg = cost_stats.groupby('strategy')['cost_usd'].agg(['mean', 'std']).reset_index()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Memory Compression Strategies Trade-offs', fontsize=16)
    
    strategies = acc_stats['strategy']
    
    sns.barplot(ax=axes[0, 0], data=df, x='strategy', y='accuracy', capsize=0.1, errorbar='sd')
    axes[0, 0].set_title('Mean Accuracy (± std)')
    axes[0, 0].set_ylim(0, 1.05)
    
    sns.barplot(ax=axes[0, 1], data=df, x='strategy', y='latency_ms', capsize=0.1, errorbar='sd')
    axes[0, 1].set_title('Mean Latency per Query (ms)')
    
    sns.barplot(ax=axes[1, 0], data=cost_stats, x='strategy', y='cost_usd', capsize=0.1, errorbar='sd')
    axes[1, 0].set_title('Total Cost per 20-Q Run (USD)')
    
    heatmap_data = pd.DataFrame({'strategy': strategies})
    heatmap_data['Accuracy'] = acc_stats['mean']
    heatmap_data['Latency'] = lat_stats['mean']
    heatmap_data['Cost'] = cost_agg['mean']
    heatmap_data.set_index('strategy', inplace=True)
    
    for col in heatmap_data.columns:
        min_val = heatmap_data[col].min()
        max_val = heatmap_data[col].max()
        if max_val > min_val:
            if col == 'Accuracy':
                heatmap_data[col] = (heatmap_data[col] - min_val) / (max_val - min_val)
            else:
                heatmap_data[col] = (max_val - heatmap_data[col]) / (max_val - min_val)
        else:
            heatmap_data[col] = 1.0
            
    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", ax=axes[1, 1], vmin=0, vmax=1)
    axes[1, 1].set_title('Normalized Performance (1.0 is Best)')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('results/dashboard.png', dpi=300)
    print("Dashboard saved to results/dashboard.png")

if __name__ == "__main__":
    visualize()
