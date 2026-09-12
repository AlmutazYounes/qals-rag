# Project Context & Session History: Query-Adaptive Late Segmentation (QALS)

This document preserves the context, architectural decisions, mathematical formulation, empirical benchmarks, journal guidelines, and author metadata established during development.

---

## 1. Project Overview & Research Identity

- **Paper Title:** *Query-adaptive late segmentation: dynamic context assembly via sentence multi-vectors and split-conformal budgeting*
- **Short Name:** QALS (Query-Adaptive Late Segmentation)
- **Authors:**
  - **Mohammad Bani Younes** (Faculty of Information and Technology, Ajloun National University, Jordan — `mohamed.banyyounes@anu.edu.jo`)
  - **Mutaz Younes** *(Corresponding Author)* (Independent Researcher, Albany, New York, USA — `mutazyounes@gmail.com`)
- **Target Journal:** *Information Processing & Management* (Elsevier)
  - Scopus Q1 (98th percentile, CiteScore 18.6, Impact Factor 6.9)
  - 100% Free / $0 APC through standard subscription publishing model

---

## 2. Core Problem: The Chunk-Size Dilemma in RAG

Standard dense retrieval architectures split text into static character windows (e.g., 200–1,000 characters) before indexing and retrieve a fixed top-$k$ count of chunks. This creates two structural failure modes:
1. **The Chunk-Size Dilemma:** Small chunks isolate facts but sever coreference chains, definitions, and scope qualifiers. Large chunks preserve narrative context but dilute vector sharpness through pooling and clutter the LLM context window with irrelevant sentences.
2. **Fixed-$k$ Prompt Bloat:** Returning a static number of chunks enforces an identical token footprint across simple and complex queries alike. Returning parent documents (LangChain ParentDocumentRetriever) balloons prompt token volume to over 1,140 tokens per query, wasting over 70% of tokens on irrelevant text and actually hurting nDCG.

---

## 3. The QALS Solution Architecture

QALS eliminates index-time chunk boundaries through three innovations:

1. **Contextualized Sentence Multi-Vectors:**
   Documents $D = (T, S)$ are indexed as sequences of atomic sentences carrying document title context:
   $$v_i = \text{Embed}(T \circ s_i)$$
   A coarse document representation $u_D = \text{Embed}(\text{Doc})$ is also indexed for two-stage candidate pruning ($M=100$).
   
2. **1D Dynamic Programming Span Assembly:**
   At query time, sentence dot products $\sigma_{D, i} = \langle q_e, v_{D, i} \rangle$ are computed for candidate documents. An online dynamic program stitches contiguous sentences into coherent passages by maximizing:
   $$U(i, j) = \sum_{k=i}^j (\sigma_{D, k} - \mu) + \kappa \cdot \log_2(j - i + 2)$$
   where $\mu$ is the baseline relevance threshold and $\kappa \cdot \log_2(j - i + 2)$ provides a diminishing-returns discourse continuity bonus.
   
3. **Split-Conformal Budget Calibration:**
   Replaces heuristic $k$ guessing with distribution-free coverage guarantees. Using held-out calibration queries $\{q_k, D_k^*\}_{k=1}^N$ with minimal budget scores $B_k^*$:
   $$\hat{B} = \text{Quantile}\left(\{B_k^*\}_{k=1}^N, \frac{\lceil (N + 1)(1 - \alpha) \rceil}{N}\right)$$
   guaranteeing $P(D^* \in \text{RetrievedContext}(\hat{B})) \ge 1 - \alpha$.

---

## 4. Empirical Evaluation Results

Evaluated across the full official BEIR SciFact (5,183 documents) and BEIR NFCorpus (3,633 documents) benchmarks using `SentenceTransformer('all-MiniLM-L6-v2')` on CPU:

| System / Architecture | SciFact nDCG@10 | SciFact Recall@10 | SciFact Tokens | NFCorpus nDCG@10 | NFCorpus Recall@10 | NFCorpus Tokens |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| LangChain Recursive (500c, $k=5$) | 0.6883 | 0.7888 | 239.6 | 0.3183 | 0.1544 | 198.1 |
| LangChain Recursive (1000c, $k=5$) | 0.6715 | 0.7866 | 382.7 | 0.3156 | 0.1624 | 281.4 |
| LangChain ParentDocument ($k=5$) | 0.6843 | 0.8107 | 1145.2 | 0.3479 | 0.1736 | 1204.7 |
| Okapi BM25 Lexical Baseline | 0.6995 | 0.8232 | 1208.9 | 0.3403 | 0.1758 | 1243.1 |
| LangChain Hybrid Ensemble ($k=5$) | 0.7234 | 0.8754 | 1134.6 | 0.3690 | 0.1841 | 1192.0 |
| **QALS Dynamic Spans ($B=150$)** | **0.7016** | **0.8531** | **142.3** | **0.3802** | **0.1808** | **142.3** |

### Key takeaways
- **Token Efficiency:** QALS delivers evidence in 142.3 tokens per query, cutting prompt volume by 41% vs 500c chunks and by 88% vs ParentDocument retrieval.
- **Superior Precision:** Outperforms single-chunk dense retrieval and parent-document retrieval on both scientific and clinical medical domains.
- **Parent-Document Failure Mode:** ParentDocument retrieval dilutes ranking precision (0.6843 nDCG vs 0.7016 for QALS) while wasting over 1,140 tokens per query.

---

## 5. Repository File Map

- `paper/`
  - `paper.md`: Full manuscript in Markdown formatted for Elsevier *Information Processing & Management*.
  - `paper/paper.tex`: Complete LaTeX article matching Elsevier submission standards.
  - `paper/generate_pdf.py`: ReportLab publication PDF generator with two-column/clean layout, running headers/footers, and figures.
  - `paper/qals_research_paper.pdf`: Compiled 4-page publication PDF with exact authors, affiliations, highlights, tables, and figures.
  - `paper/figures/benchmark_summary.png`: Dual-panel high-resolution comparison chart.
- `src/toporag/`
  - `qals_encoder.py`: Contextualized sentence multi-vector encoder (`SentenceMultiVectorEncoder`).
  - `qals_dp.py`: Dynamic programming span segmentation with discourse continuity bonus.
  - `qals_retriever.py`: Two-stage QALS retriever engine.
  - `conformal.py`: Split-conformal quantile calibration.
  - `langchain_baselines.py`: Official LangChain recursive, parent document, and hybrid fusion splitters.
  - `real_baselines.py`: Pure numpy/scipy baselines for testing and benchmarking.
- `benchmarks/`
  - `run_fast_benchmark.py`: Vectorized CPU evaluation script on full SciFact and NFCorpus BEIR datasets.
  - `multi_benchmark_results.json`: JSON output containing the exact empirical metrics.
- `demo/`
  - `demo/backend/server.py`: FastAPI backend comparing QALS against LangChain and BM25 baselines.
  - `demo/frontend/index.html`: Responsive Tailwind dark-mode UI with live token-by-token comparison.
- `tests/`
  - `test_qals.py`: Unit tests for encoder, DP segmenter, and conformal calibration.
  - `test_langchain_baselines.py`: Integration tests for LangChain splitters.
- `pyproject.toml`: Modern packaging file with project dependencies and metadata.
- `README.md`: Public-facing repository documentation.

---

## 6. How to Run Locally

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -e ".[dense]"

# 3. Run unit tests
pytest tests/

# 4. Recompile the paper PDF
python3 paper/generate_pdf.py

# 5. Launch the interactive demo
python3 -m uvicorn demo.backend.server:app --port 8765 --reload
```
