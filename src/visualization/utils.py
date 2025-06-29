import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

def load_results(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def plot_metrics_by_k(results, metric_name, title):
    k_values = sorted([int(k) for k in results['metrics'][metric_name].keys()])
    values = [results['metrics'][metric_name][str(k)] for k in k_values]

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, values, marker='o', linestyle='-', linewidth=2)
    plt.title(f'{title} vs K')
    plt.xlabel('K')
    plt.ylabel(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(k_values)
    plt.tight_layout()
    return plt.gcf()

def plot_all_metrics(results, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        'hit_rate': 'Hit Rate',
        'map': 'Mean Average Precision (MAP)',
        'ndcg': 'Normalized Discounted Cumulative Gain (NDCG)'
    }

    for metric, title in metrics.items():
        fig = plot_metrics_by_k(results, metric, title)
        fig.savefig(save_dir / f'{metric}_vs_k.png')
        plt.close(fig)

    plt.figure(figsize=(12, 8))
    k_values = sorted([int(k) for k in results['metrics']['hit_rate'].keys()])
    for metric, title in metrics.items():
        values = [results['metrics'][metric][str(k)] for k in k_values]
        plt.plot(k_values, values, marker='o', label=title)

    plt.title('All Metrics vs K')
    plt.xlabel('K')
    plt.ylabel('Score')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.xticks(k_values)
    plt.tight_layout()
    plt.savefig(save_dir / 'all_metrics_vs_k.png')
    plt.close()

def plot_metric_distribution(results, save_dir):
    save_dir = Path(save_dir)

    user_metrics = []
    for user_id, metrics in results['user_metrics'].items():
        for k, values in metrics.items():
            user_metrics.append({
                'user_id': user_id,
                'k': int(k),
                'hit_rate': values['hit_rate'],
                'map': values['map'],
                'ndcg': values['ndcg']
            })

    df = pd.DataFrame(user_metrics)
    metrics = ['hit_rate', 'map', 'ndcg']
    k_values = sorted(df['k'].unique())

    for metric in metrics:
        plt.figure(figsize=(15, 5))
        for i, k in enumerate(k_values, 1):
            plt.subplot(1, len(k_values), i)
            data = df[df['k'] == k][metric]
            sns.histplot(data=data, bins=30)
            plt.title(f'{metric.upper()} at k={k}')
            plt.xlabel(metric)
            plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(save_dir / f'{metric}_distribution.png')
        plt.close()
