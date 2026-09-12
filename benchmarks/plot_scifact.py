"""
Generates publication plots for the BEIR SciFact benchmark results.
Shows:
1. Token Efficiency: Signal-to-Noise Ratio vs Average Delivered Tokens
2. Retrieval Quality: NDCG and MRR under strict token budgeting
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import json
import os


def generate_scifact_plots():
    os.makedirs("paper/figures", exist_ok=True)
    with open("benchmarks/scifact_results.json", "r") as f:
        data = json.load(f)

    models = list(data.keys())
    # Short names for cleaner plotting
    short_names = [
        "BM25\n(k=5)",
        "Flat Dense\n(Fixed 500c)",
        "Parent-Doc\n(Child->Parent)",
        "QALS\n(B=150)",
        "QALS\n(Conformal)",
    ]

    tokens_delivered = [data[m]["Avg Delivered Tokens"] for m in models]
    snr_scores = [data[m]["Signal-to-Noise Ratio (Gold/Total Tokens)"] for m in models]
    ndcg_scores = [data[m]["NDCG"] * 100 for m in models]
    latencies = [data[m]["Avg Latency (ms)"] for m in models]

    # Plot 1: Token Efficiency (Signal-to-Noise Ratio)
    fig, ax1 = plt.subplots(figsize=(8.5, 4.5))
    colors = ["#94a3b8", "#38bdf8", "#818cf8", "#10b981", "#059669"]
    bars = ax1.bar(short_names, snr_scores, color=colors, width=0.55, edgecolor="#0f172a", linewidth=1.2)

    ax1.set_ylabel("Signal-to-Noise Ratio (%) [Gold / Total Delivered Tokens]", fontsize=10, fontweight="bold")
    ax1.set_title("Token Efficiency on BEIR SciFact: Gold Evidence Concentration", fontsize=12, fontweight="bold", pad=12)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.set_ylim(0, 105)

    for bar, tok in zip(bars, tokens_delivered):
        h = bar.get_height()
        ax1.annotate(
            f"{h:.1f}%\n({int(tok)} tok)",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig("paper/figures/scifact_snr.png", dpi=300)
    plt.close()

    # Plot 2: Pareto Efficiency: NDCG vs Delivered Tokens
    fig, ax2 = plt.subplots(figsize=(8, 4.5))
    scatter_colors = ["#64748b", "#0284c7", "#6366f1", "#10b981", "#047857"]
    
    for i, name in enumerate(short_names):
        ax2.scatter(
            tokens_delivered[i],
            ndcg_scores[i],
            color=scatter_colors[i],
            s=160,
            edgecolors="#0f172a",
            linewidth=1.5,
            zorder=4,
        )
        ax2.annotate(
            name.replace("\n", " "),
            (tokens_delivered[i], ndcg_scores[i]),
            xytext=(8, -2),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold",
        )

    ax2.set_xlabel("Average Delivered Tokens per Query", fontsize=10, fontweight="bold")
    ax2.set_ylabel("NDCG Score (%)", fontsize=10, fontweight="bold")
    ax2.set_title("Pareto Efficiency: NDCG vs Context Token Footprint on BEIR SciFact", fontsize=12, fontweight="bold", pad=12)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("paper/figures/scifact_pareto.png", dpi=300)
    plt.close()

    print("SciFact benchmark figures successfully created in paper/figures/")


if __name__ == "__main__":
    generate_scifact_plots()
