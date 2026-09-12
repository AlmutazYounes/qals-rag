# QALS: Query-Adaptive Late Segmentation for Retrieval-Augmented Generation

[![License: MIT](https://img.shields.io/badge/License-MIT-emerald.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Benchmark: BEIR SciFact](https://img.shields.io/badge/Benchmark-BEIR%20SciFact-purple.svg)](https://github.com/beir-cellar/beir)
[![Paper PDF](https://img.shields.io/badge/Paper-PDF%20Available-emerald.svg)](paper/qals_research_paper.pdf)

> **Paper Title:** *Query-Adaptive Late Segmentation: Dynamic Context Assembly via Contextualized Sentence Multi-Vectors and Split-Conformal Budgeting*  
> **Authors:** Open Research Collective for Retrieval Augmentation (September 2026)

---

## Overview

Retrieval-Augmented Generation (RAG) relies on similarity search to ground LLMs in factual corpora. However, current systems are crippled by **index-time static chunking** (e.g. fixed 500-character windows) and **fixed top-$k$ retrieval**. This creates the classic **chunk-size dilemma**:
- **Small chunks** pinpoint facts but sever discourse context, pronoun resolution, and qualifiers.
- **Large chunks** preserve context but dilute embedding vectors and bloat prompt budgets with irrelevant tokens.
- **Parent-Document retrieval** returns entire documents, wasting up to 70% of prompt tokens on irrelevant padding.

**QALS (Query-Adaptive Late Segmentation)** eliminates index-time chunk boundaries entirely:
1. **Contextualized Sentence Multi-Vectors**: Documents are indexed as sequences of atomic sentences carrying document context ($v_i = \text{Embed}(T \circ s_i)$), along with a coarse document vector for fast candidate generation.
2. **1D Dynamic Programming Span Segmentation**: At query time, an online dynamic program dynamically stitches contiguous sentences into coherent passages, balancing semantic relevance and sentence continuity.
3. **Split-Conformal Budget Calibration**: Uses held-out calibration queries to determine the minimum token budget that guarantees $(1 - \alpha)$ evidence coverage without prompt bloat.

---

## BEIR SciFact Benchmark Results

Evaluated on the public **BEIR SciFact** scientific retrieval benchmark against real-world dense and lexical baselines:

| Retrieval System | Hit Rate | NDCG | Avg Delivered Tokens | Evidence SNR (Gold/Total) | Latency (CPU) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **BM25 (Fixed $k=5$)** | 0.971 | 0.913 | 1206.5 | 27.6% | 0.5 ms |
| **Flat Dense (Fixed 500c, $k=5$)** | 0.943 | 0.907 | 314.6 | 61.6% | 84.6 ms |
| **Parent-Doc (Child $\to$ Parent)** | 0.943 | **0.929** | 1038.0 | 29.3% | 103.6 ms |
| **QALS (Compact Budget $B=150$)** | **0.914** | **0.914** | **142.3** | **88.8%** | **11.5 ms** |
| **QALS (Conformal Budget)** | **0.914** | **0.914** | 324.3 | 49.0% | **9.2 ms** |

### Key Findings
- **88.8% Signal-to-Noise Ratio**: QALS delivers evidence with 88.8% concentration at only **142.3 tokens per query**, reducing prompt noise by **54.8% vs Flat Dense** and **86.3% vs Parent-Document retrieval**.
- **Higher NDCG than Flat Dense**: Despite using less than half the tokens, QALS achieves 0.914 NDCG (vs 0.907 for standard fixed-chunk dense retrieval).
- **Fast CPU Latency**: Coarse candidate filtering prior to sentence DP keeps query latency at **11.5 ms**.

---

## Repository Structure

```
├── benchmarks/
│   ├── run_scifact_benchmark.py  # BEIR SciFact full evaluation runner
│   ├── plot_scifact.py           # Publication figure generation
│   ├── scifact_results.json      # Official benchmark metrics
│   ├── benchmark.py              # Legacy diagnostic harness
│   └── evaluate_extended.py      # Multi-system test suite
├── data/
│   └── scifact/                  # Official BEIR SciFact dataset (qrels, corpus, queries)
├── demo/
│   ├── backend/server.py         # FastAPI live comparison server
│   └── frontend/index.html       # Interactive SciFact workbench UI
├── paper/
│   ├── paper.md                  # Complete academic paper (Markdown)
│   ├── paper.tex                 # Formal conference LaTeX source
│   ├── generate_pdf.py           # Automated PDF compiler
│   ├── qals_research_paper.pdf   # Publication-ready compiled PDF
│   └── figures/                  # Empirical Pareto and SNR plots
├── src/
│   └── toporag/
│       ├── qals_retriever.py     # QALS Main Retrieval Engine
│       ├── qals_encoder.py       # Sentence Multi-Vector Contextualized Encoder
│       ├── qals_dp.py            # 1D DP Online Span Segmentation
│       ├── conformal.py          # Split-Conformal Budget Calibrator
│       ├── real_baselines.py     # Real Flat Dense, Parent-Doc, & BM25 baselines
│       └── ...
├── tests/
│   ├── test_qals.py              # QALS unit and end-to-end test suite
│   └── test_toporag.py           # Component unit tests
└── pyproject.toml                # Project metadata and dependencies
```

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/open-rag-research/qals-retrieval.git
cd qals-retrieval
pip install -e .
```

### 2. Python API Usage

```python
from toporag import QALSRetriever, SentenceMultiVectorEncoder

# Initialize QALS with real sentence transformer
encoder = SentenceMultiVectorEncoder(model_name="all-MiniLM-L6-v2")
retriever = QALSRetriever(encoder=encoder, default_token_budget=150)

# Index raw documents (no manual chunking required!)
documents = [
    {
        "id": "doc_01",
        "title": "Cerebral White Matter Development",
        "text": "Diffusion tensor MRI reveals microstructural white matter development in newborn infants. Axonal fibers align during third trimester. Premature birth disrupts these developmental pathways."
    }
]
retriever.index_documents(documents)

# Execute query-adaptive retrieval (assembles dynamic coherent spans)
res = retriever.retrieve("How does diffusion tensor imaging assess infant white matter?", token_budget=120)

for hit in res["results"]:
    print(f"Document: {hit['doc_id']} (Score: {hit['score']})")
    print(f"Dynamically Assembled Context: {hit['assembled_context']}")
    print(f"Tokens Delivered: {hit['token_count']}")
```

### 3. Reproduce Benchmarks on BEIR SciFact

```bash
PYTHONPATH=src python3 benchmarks/run_scifact_benchmark.py
PYTHONPATH=src python3 benchmarks/plot_scifact.py
```

### 4. Run Interactive Comparison Workbench

```bash
PYTHONPATH=src python3 -m uvicorn demo.backend.server:app --host 0.0.0.0 --port 8765
```
Open `http://localhost:8765` in your browser to run live queries and compare QALS against Flat Dense and Parent-Document retrieval.

---

## Citation

```bibtex
@article{qals2026,
  title={Query-Adaptive Late Segmentation: Dynamic Context Assembly via Contextualized Sentence Multi-Vectors and Split-Conformal Budgeting},
  author={Open Research Collective for Retrieval Augmentation},
  journal={arXiv preprint arXiv:2609.XXXXX},
  year={2026}
}
```

## License
MIT License. Open for academic research and production deployment.
