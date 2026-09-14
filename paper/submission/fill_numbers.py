#!/usr/bin/env python3
"""Fill manuscript macros and draw Figure_1 / Figure_2 from benchmark JSON.

Primary source: benchmarks/ipm_benchmark_results.json
Optional overlays (leave TBD macros unchanged when missing):
  benchmarks/ablation_results.json
  benchmarks/rag_generation_results.json
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
JSON_PATH = ROOT / "benchmarks" / "ipm_benchmark_results.json"
ABLATION_JSON = ROOT / "benchmarks" / "ablation_results.json"
RAG_JSON = ROOT / "benchmarks" / "rag_generation_results.json"
TEX_PATH = Path(__file__).resolve().parent / "manuscript.tex"
OUT = Path(__file__).resolve().parent

KEYS = {
    "scifact": {
        "LangChain Recursive 500c": ('ScifactLcNdcg', None, 'ScifactLcTok'),
        "LangChain Recursive 1000c": ('ScifactLclongNdcg', None, 'ScifactLclongTok'),
        "LangChain ParentDocument": ('ScifactParentNdcg', None, 'ScifactParentTok'),
        "BM25": ('ScifactBmNdcg', None, 'ScifactBmTok'),
        "LangChain Hybrid RRF": ('ScifactHybridNdcg', None, 'ScifactHybridTok'),
        "Adaptive-k (LC500 gap)": ('ScifactAdaptkNdcg', 'ScifactAdaptkRec', 'ScifactAdaptkTok'),
        "QALS B=150": ('ScifactQalsNdcg', 'ScifactQalsRec', 'ScifactQalsTok'),
    },
    "nfcorpus": {
        "LangChain Recursive 500c": ('NfLcNdcg', None, 'NfLcTok'),
        "LangChain Recursive 1000c": ('NfLclongNdcg', None, 'NfLclongTok'),
        "LangChain ParentDocument": ('NfParentNdcg', None, 'NfParentTok'),
        "BM25": ('NfBmNdcg', None, 'NfBmTok'),
        "LangChain Hybrid RRF": ('NfHybridNdcg', None, 'NfHybridTok'),
        "Adaptive-k (LC500 gap)": ('NfAdaptkNdcg', 'NfAdaptkRec', 'NfAdaptkTok'),
        "QALS B=150": ('NfQalsNdcg', 'NfQalsRec', 'NfQalsTok'),
    },
}

ABLATION_MACROS = {
    "B=50": ("AblatBfiftyNdcg", "AblatBfiftyTok"),
    "B=100": ("AblatBhundredNdcg", "AblatBhundredTok"),
    "B=150": ("AblatBtonefiftyNdcg", "AblatBtonefiftyTok"),
    "B=200": ("AblatBtwohundredNdcg", "AblatBtwohundredTok"),
    "kappa=0.00": ("AblatKappaZeroNdcg", "AblatKappaZeroTok"),
    "M=20": ("AblatMtwentyNdcg", "AblatMtwentyTok"),
}

RAG_SYSTEM_MACROS = {
    "QALS": ("RagQalsAcc", "RagQalsFaith", "RagQalsTok"),
    "Recursive500": ("RagLcAcc", "RagLcFaith", "RagLcTok"),
    "ParentDocument": ("RagParentDocAcc", "RagParentDocFaith", "RagParentDocTok"),
    "HybridRRF": ("RagHybridAcc", "RagHybridFaith", "RagHybridTok"),
}

# Manuscript Table tab:reader currently cites DeBERTa NLI numbers.
NLI_READER_JSON = ROOT / "benchmarks" / "reader_nli_results.json"
NLI_SYSTEM_MACROS = {
    "Recursive500_k5": ("RagLcAcc", "RagLcFaith", "RagLcTok"),
    "AdaptiveK_Recursive500": ("RagParentAcc", "RagParentFaith", "RagParentTok"),
    "Hybrid_docs_B150": ("RagHybridAcc", "RagHybridFaith", "RagHybridTok"),
    "QALS_B150": ("RagQalsAcc", "RagQalsFaith", "RagQalsTok"),
}

# Separate macros for the HF / oracle generation eval JSON.
HF_SYSTEM_MACROS = {
    "QALS": ("RagHfQalsAcc", "RagHfQalsFaith", "RagHfQalsTok"),
    "Recursive500": ("RagHfLcAcc", "RagHfLcFaith", "RagHfLcTok"),
    "ParentDocument": ("RagHfParentAcc", "RagHfParentFaith", "RagHfParentTok"),
    "HybridRRF": ("RagHfHybridAcc", "RagHfHybridFaith", "RagHfHybridTok"),
}


def fmt(v, ndigits):
    return f"{float(v):.{ndigits}f}"


def patch_macro(text, name, value):
    pattern = rf"(\\newcommand{{\\{name}}}{{)[^}}]*}}"
    if not re.search(pattern, text):
        # Insert before \begin{document} if the macro is new.
        text = text.replace(
            "\\begin{document}",
            f"\\newcommand{{\\{name}}}{{{value}}}\n\\begin{{document}}",
            1,
        )
        return text
    return re.sub(pattern, rf"\g<1>{value}}}", text)


def plot(data):
    labels = [
        ("Recursive 500c", "LangChain Recursive 500c"),
        ("Recursive 1000c", "LangChain Recursive 1000c"),
        ("Parent", "LangChain ParentDocument"),
        ("BM25", "BM25"),
        ("Hybrid RRF", "LangChain Hybrid RRF"),
        ("Adaptive-k", "Adaptive-k (LC500 gap)"),
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
        "Adaptive-k": "#c2410c",
    }
    for ax, corp, title in zip(axes, ("scifact", "nfcorpus"), ("SciFact", "NFCorpus")):
        block = data.get(corp, {})
        for short, key in labels:
            row = block.get(key)
            if not row:
                continue
            ax.scatter(row["tokens"], row["ndcg"], s=70, color=colors[short], zorder=3)
            ax.annotate(short, (row["tokens"], row["ndcg"]), textcoords="offset points", xytext=(6, 4), fontsize=8)
        ax.set_xlabel("Mean prompt tokens")
        ax.set_ylabel("nDCG@10")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "Figure_1.png", dpi=200)
    plt.close(fig)

    present = [(s, k) for s, k in labels if data.get("scifact", {}).get(k) and data.get("nfcorpus", {}).get(k)]
    fig, ax = plt.subplots(figsize=(10.2, 4.0))
    x = np.arange(len(present))
    w = 0.38
    sc = [data["scifact"][k]["tokens"] for _, k in present]
    nf = [data["nfcorpus"][k]["tokens"] for _, k in present]
    ax.bar(x - w / 2, sc, w, label="SciFact", color="#0f766e")
    ax.bar(x + w / 2, nf, w, label="NFCorpus", color="#b45309")
    ax.set_xticks(x)
    ax.set_xticklabels([s for s, _ in present], rotation=20, ha="right")
    ax.set_ylabel("Mean prompt tokens")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "Figure_2.png", dpi=200)
    plt.close(fig)


def pick_rag_backend(block: dict):
    """Prefer API backends, then HF instruct, then oracle, then extractive."""
    backends = block.get("backends") or {}
    for name in ("openai", "anthropic", "hf_instruct", "oracle_answerability", "extractive_minilm"):
        if name in backends:
            return name, backends[name]
    if backends:
        name = next(iter(backends))
        return name, backends[name]
    return None, None


def main():
    data = json.loads(JSON_PATH.read_text())
    tex = TEX_PATH.read_text()

    for corp, mapping in KEYS.items():
        for sysname, macros in mapping.items():
            row = data.get(corp, {}).get(sysname)
            if not row:
                continue
            tex = patch_macro(tex, macros[0], fmt(row["ndcg"], 4))
            if macros[1]:
                tex = patch_macro(tex, macros[1], fmt(row.get("recall", 0), 4))
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

    ablations = {}
    if ABLATION_JSON.exists():
        ablations = json.loads(ABLATION_JSON.read_text()).get("scifact", {})
    if not ablations:
        ablations = data.get("scifact", {}).get("ablations") or {}
    for label, macros in ABLATION_MACROS.items():
        row = ablations.get(label)
        if not row:
            continue
        tex = patch_macro(tex, macros[0], fmt(row["ndcg"], 4))
        tex = patch_macro(tex, macros[1], fmt(row["tokens"], 1))

    if RAG_JSON.exists():
        rag = json.loads(RAG_JSON.read_text())
        scifact = rag.get("scifact") or {}
        backend_name, backend = pick_rag_backend(scifact)
        if backend_name:
            tex = patch_macro(tex, "RagHfBackendName", backend_name.replace("_", "\\_"))
            for sysname, macros in HF_SYSTEM_MACROS.items():
                summary = (backend.get(sysname) or {}).get("summary") or {}
                if not summary:
                    continue
                acc = summary.get("accuracy")
                if acc is not None:
                    tex = patch_macro(tex, macros[0], fmt(acc, 4))
                faith = summary.get("mean_faithfulness")
                if faith is not None:
                    tex = patch_macro(tex, macros[1], fmt(faith, 4))
                tok = summary.get("mean_prompt_tokens")
                if tok is not None:
                    tex = patch_macro(tex, macros[2], fmt(tok, 1))

    # Prefer the DeBERTa NLI reader JSON for the manuscript Rag* table when present.
    if NLI_READER_JSON.exists():
        nli = json.loads(NLI_READER_JSON.read_text())
        tex = patch_macro(tex, "RagBackendName", "nli\\_deberta\\_v3\\_xsmall")
        systems = nli.get("systems") or {}
        for sysname, macros in NLI_SYSTEM_MACROS.items():
            row = systems.get(sysname) or {}
            if not row:
                continue
            if row.get("label_accuracy") is not None:
                tex = patch_macro(tex, macros[0], fmt(row["label_accuracy"], 4))
            # Manuscript "Faith" column is non-neutral rate for the NLI reader.
            faith = row.get("non_neutral_rate")
            if faith is None and row.get("neutral_rate") is not None:
                faith = 1.0 - float(row["neutral_rate"])
            if faith is not None:
                tex = patch_macro(tex, macros[1], fmt(faith, 4))
            if row.get("mean_prompt_tokens") is not None:
                tex = patch_macro(tex, macros[2], fmt(row["mean_prompt_tokens"], 1))
    elif RAG_JSON.exists():
        rag = json.loads(RAG_JSON.read_text())
        scifact = rag.get("scifact") or {}
        backend_name, backend = pick_rag_backend(scifact)
        if backend_name:
            tex = patch_macro(tex, "RagBackendName", backend_name.replace("_", "\\_"))
            for sysname, macros in RAG_SYSTEM_MACROS.items():
                summary = (backend.get(sysname) or {}).get("summary") or {}
                if not summary:
                    continue
                acc = summary.get("accuracy")
                if acc is not None:
                    tex = patch_macro(tex, macros[0], fmt(acc, 4))
                faith = summary.get("mean_faithfulness")
                if faith is not None:
                    tex = patch_macro(tex, macros[1], fmt(faith, 4))
                tok = summary.get("mean_prompt_tokens")
                if tok is not None:
                    tex = patch_macro(tex, macros[2], fmt(tok, 1))

    TEX_PATH.write_text(tex)
    plot(data)
    print("Updated manuscript macros and figures.")


if __name__ == "__main__":
    main()
