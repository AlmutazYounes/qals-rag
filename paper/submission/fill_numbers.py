#!/usr/bin/env python3
"""Fill manuscript macros and draw Figure_1 / Figure_2 from ipm_benchmark_results.json."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
JSON_PATH = ROOT / "benchmarks" / "ipm_benchmark_results.json"
TEX_PATH = Path(__file__).resolve().parent / "manuscript.tex"
OUT = Path(__file__).resolve().parent

KEYS = {
    "scifact": {
        "QALS B=150": ("ScifactQalsNdcg", "ScifactQalsRec", "ScifactQalsTok"),
        "LangChain Recursive 500c": ("ScifactLcNdcg", None, "ScifactLcTok"),
        "LangChain Recursive 1000c": ("ScifactLclongNdcg", None, "ScifactLclongTok"),
        "LangChain ParentDocument": ("ScifactParentNdcg", None, "ScifactParentTok"),
        "LangChain Hybrid RRF": ("ScifactHybridNdcg", None, "ScifactHybridTok"),
        "BM25": ("ScifactBmNdcg", None, "ScifactBmTok"),
    },
    "nfcorpus": {
        "QALS B=150": ("NfQalsNdcg", "NfQalsRec", "NfQalsTok"),
        "LangChain Recursive 500c": ("NfLcNdcg", None, "NfLcTok"),
        "LangChain Recursive 1000c": ("NfLclongNdcg", None, "NfLclongTok"),
        "LangChain ParentDocument": ("NfParentNdcg", None, "NfParentTok"),
        "LangChain Hybrid RRF": ("NfHybridNdcg", None, "NfHybridTok"),
        "BM25": ("NfBmNdcg", None, "NfBmTok"),
    },
}


def fmt(v, ndigits):
    return f"{float(v):.{ndigits}f}"


def patch_macro(text, name, value):
    import re
    return re.sub(
        rf"(\\newcommand{{\\{name}}}{{)[^}}]*}}",
        rf"\g<1>{value}}}",
        text,
    )


def plot(data):
    labels = [
        ("Recursive 500c", "LangChain Recursive 500c"),
        ("Recursive 1000c", "LangChain Recursive 1000c"),
        ("Parent", "LangChain ParentDocument"),
        ("BM25", "BM25"),
        ("Hybrid RRF", "LangChain Hybrid RRF"),
        ("QALS", "QALS B=150"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    colors = {
        "QALS": "#0f766e",
        "Hybrid RRF": "#b45309",
        "Parent": "#7c3aed",
        "BM25": "#334155",
        "Recursive 500c": "#2563eb",
        "Recursive 1000c": "#64748b",
    }
    for ax, corp, title in zip(axes, ("scifact", "nfcorpus"), ("SciFact", "NFCorpus")):
        block = data[corp]
        for short, key in labels:
            row = block[key]
            ax.scatter(row["tokens"], row["ndcg"], s=70, color=colors[short], zorder=3)
            ax.annotate(short, (row["tokens"], row["ndcg"]), textcoords="offset points", xytext=(6, 4), fontsize=8)
        ax.set_xlabel("Mean prompt tokens")
        ax.set_ylabel("nDCG@10")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "Figure_1.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.2, 4.0))
    x = np.arange(len(labels))
    w = 0.38
    sc = [data["scifact"][k]["tokens"] for _, k in labels]
    nf = [data["nfcorpus"][k]["tokens"] for _, k in labels]
    ax.bar(x - w / 2, sc, w, label="SciFact", color="#0f766e")
    ax.bar(x + w / 2, nf, w, label="NFCorpus", color="#b45309")
    ax.set_xticks(x)
    ax.set_xticklabels([s for s, _ in labels], rotation=20, ha="right")
    ax.set_ylabel("Mean prompt tokens")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "Figure_2.png", dpi=200)
    plt.close(fig)


def main():
    data = json.loads(JSON_PATH.read_text())
    tex = TEX_PATH.read_text()
    for corp, mapping in KEYS.items():
        for sysname, macros in mapping.items():
            row = data[corp][sysname]
            tex = patch_macro(tex, macros[0], fmt(row["ndcg"], 4))
            if macros[1]:
                tex = patch_macro(tex, macros[1], fmt(row["recall"], 4))
            tex = patch_macro(tex, macros[2], fmt(row["tokens"], 1))
    conf = data.get("scifact", {}).get("conformal", {})
    a = conf.get("alpha=0.1", {})
    b = conf.get("alpha=0.2", {})
    if a:
        tex = patch_macro(tex, "ConfBhatA", str(a.get("B_hat", "TBD")))
        tex = patch_macro(tex, "ConfCovA", fmt(a.get("coverage", 0), 3))
    if b:
        tex = patch_macro(tex, "ConfBhatB", str(b.get("B_hat", "TBD")))
        tex = patch_macro(tex, "ConfCovB", fmt(b.get("coverage", 0), 3))
    TEX_PATH.write_text(tex)
    plot(data)
    print("Updated manuscript macros and figures.")


if __name__ == "__main__":
    main()
