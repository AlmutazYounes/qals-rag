"""
Generate publication-quality benchmark plots comparing TopoRAG against baselines.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import json
import os


def generate_plots():
    os.makedirs("paper/figures", exist_ok=True)
    with open("benchmarks/extended_results.json", "r") as f:
        data = json.load(f)

    models = list(data.keys())
    hit_rates = [data[m]["Hit Rate"] * 100 for m in models]
    ndcg_scores = [data[m]["NDCG"] for m in models]
    hub_ginis = [data[m]["Hub Gini"] for m in models]

    # Plot 1: Retrieval Quality (Hit Rate & NDCG)
    x = np.arange(len(models))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    rects1 = ax1.bar(x - width/2, hit_rates, width, label='Hit Rate (%)', color='#2563eb')
    rects2 = ax1.bar(x + width/2, [s * 100 for s in ndcg_scores], width, label='NDCG Score (x100)', color='#10b981')

    ax1.set_ylabel('Score / Percentage')
    ax1.set_title('Retrieval Quality: TopoRAG vs Standard Baselines')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=12, ha='right')
    ax1.legend(loc='lower right')
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("paper/figures/retrieval_quality.png", dpi=300)
    plt.close()

    # Plot 2: Hubness Concentration (Gini Coefficient)
    fig, ax2 = plt.subplots(figsize=(7, 4))
    colors = ['#94a3b8', '#f87171', '#fb923c', '#0ea5e9']
    bars = ax2.bar(models, hub_ginis, color=colors, width=0.55)
    ax2.set_ylabel('Hub Concentration (Gini Coefficient - Lower is Better)')
    ax2.set_title('Topological Fairness: Reduction in Hubness Dominance')
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig("paper/figures/hubness_reduction.png", dpi=300)
    plt.close()
    print("Benchmark figures generated in paper/figures/")


if __name__ == "__main__":
    generate_plots()
