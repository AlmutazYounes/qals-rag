# TopoRAG: Beyond Static Similarity in Retrieval-Augmented Generation

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Paper](https://img.shields.io/badge/Paper-PDF%20Available-emerald.svg)](paper/toporag_research_paper.pdf)

> **Paper Title:** *Beyond Static Similarity: Manifold-Calibrated Adaptive Multi-Granularity Retrieval for RAG*  
> **Authors:** Open Research Collective for Retrieval Augmentation (September 2026)

---

## Overview

Retrieval-Augmented Generation (RAG) relies on dense similarity search to augment LLMs with external non-parametric knowledge. Standard implementations rely on three restrictive assumptions:
1. **Fixed Sliding-Window Chunking**: Causes the chunk size dilemma (small chunks lack context; large chunks dilute relevance).
2. **Uncalibrated Inner-Product / Cosine Similarity**: Anisotropy causes high-density "hubs" near the manifold center of mass to dominate nearest-neighbor results across unrelated queries.
3. **Fixed Retrieval Depth ($k$)**: Hardcoded top-$k$ injects irrelevant noise tokens into simple factual questions and truncates complex multi-hop queries.

**TopoRAG** addresses these three bottlenecks through three interconnected mechanisms:
- **Multi-Scale Document Graph**: Segments documents into Macro (sections), Meso (paragraphs), and Micro (sentences). Vector search operates on high-specificity micro chunks; context delivery expands upward to deduplicated meso parents.
- **Riemannian Manifold Calibration**: Estimates local neighborhood density $r_k(x)$ and empirical hubness $H(x)$, applying dynamic inverted penalization:
  $$S_{\text{cal}}(q, x) = S_{\text{raw}}(q, x) - \lambda \cdot r_k(x) - \gamma \cdot \ln(1 + H(x))$$
- **Dynamic Knee & Score Entropy Cutoff**: Computes consecutive score drops and modulates retrieval depth based on query entropy, dynamically selecting the natural relevance elbow.

---

## Benchmark Highlights

Evaluated across a multi-domain 20-document technical challenge corpus spanning distributed systems, database internals, quantum error correction, and neural geometry:

| Retrieval System | Hit Rate | MRR | NDCG | Hub Dominance (Gini $\downarrow$) |
| :--- | :---: | :---: | :---: | :---: |
| **Flat Dense (Fixed $k=5$)** | 0.833 | 0.833 | 0.814 | 0.475 |
| **Fine-Grained Dense ($k=5$)** | 0.917 | 0.767 | 0.790 | 0.443 |
| **BM25 Lexical Baseline** | 1.000 | 0.958 | 0.969 | 0.373 |
| **TopoRAG (Proposed)** | **1.000** | **0.878** | **0.907** | **0.220** |

**Key Finding**: TopoRAG reduces hub dominance concentration by **53.7%** (Gini drops from 0.475 to 0.220), ensuring that documents compete on genuine semantic relevance rather than geometric centrality.

---

## Repository Structure

```
├── benchmarks/
│   ├── benchmark.py            # Automated benchmark evaluation harness
│   ├── evaluate_extended.py    # Extended challenge evaluation
│   ├── plot_results.py         # Publication-quality figure generation
│   ├── results.json            # Empirical benchmark metrics
│   └── extended_results.json   # Extended empirical metrics
├── demo/
│   ├── backend/
│   │   └── server.py           # FastAPI demonstration server
│   └── frontend/
│       └── index.html          # Interactive side-by-side comparison UI
├── paper/
│   ├── paper.md                # Full academic paper in Markdown
│   ├── paper.tex               # Formal two-column conference LaTeX paper
│   ├── generate_pdf.py         # Automated PDF compiler
│   ├── toporag_research_paper.pdf # Compiled publication-ready PDF
│   └── figures/                # Empirical benchmark charts
├── src/
│   └── toporag/
│       ├── __init__.py         # Package exports
│       ├── chunking.py         # Multi-scale hierarchical chunker
│       ├── calibration.py      # Manifold & hubness calibration
│       ├── dynamic_cutoff.py   # Adaptive knee & entropy cutoff
│       ├── embeddings.py       # Deterministic & dense embedder wrappers
│       ├── baselines.py        # Flat dense and BM25 baseline implementations
│       └── retriever.py        # TopoRAG main engine
├── tests/
│   └── test_toporag.py         # Comprehensive unit tests
└── pyproject.toml              # Build and dependency definition
```

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/open-rag-research/toporag.git
cd toporag
pip install -e .
```

### 2. Python API Usage

```python
from toporag import TopoRAGRetriever, HashFeatureEmbedder

# Initialize retriever
embedder = HashFeatureEmbedder(dim=256)
retriever = TopoRAGRetriever(embedder=embedder)

# Index raw documents
documents = [
    {
        "id": "doc_raft",
        "text": "The Raft consensus protocol elects a leader to replicate log entries across follower nodes."
    },
    {
        "id": "doc_geom",
        "text": "High-dimensional neural embeddings exhibit anisotropy, creating artificial hub vectors in nearest neighbor search."
    }
]
retriever.index_documents(documents)

# Execute adaptive retrieval
response = retriever.retrieve("Why do vector embeddings create hubs in similarity search?")

for hit in response["results"]:
    print(f"Context: {hit['retrieved_context']}")
    print(f"Calibrated Score: {hit['calibrated_score']}, Raw: {hit['raw_score']}")
```

### 3. Running Benchmarks

```bash
PYTHONPATH=src python3 benchmarks/evaluate_extended.py
PYTHONPATH=src python3 benchmarks/plot_results.py
```

### 4. Running the Interactive Visual Demonstrator

```bash
PYTHONPATH=src python3 -m uvicorn demo.backend.server:app --host 0.0.0.0 --port 8765
```
Open your browser at `http://localhost:8765` to compare TopoRAG side-by-side with Flat Dense and BM25 retrievers.

---

## Citation

```bibtex
@article{toporag2026,
  title={Beyond Static Similarity: Manifold-Calibrated Adaptive Multi-Granularity Retrieval for RAG},
  author={Open Research Collective for Retrieval Augmentation},
  journal={arXiv preprint arXiv:2609.XXXXX},
  year={2026}
}
```

## License
MIT License. Open for research and commercial adoption.
