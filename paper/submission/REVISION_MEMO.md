# QALS revision memo: related work, gaps, baselines, framing

Audience: next-venue rewrite after IP&M desk / EiC feedback that this stage of RAG is already crowded and that SOTA baselines, especially LLM-facing ones, plus current-year citations are missing.

Companion file: `references_upgrade.bib` (20 verified entries, mostly 2024--2026). Merge into `references.bib`; do not invent DOIs.

## What the current manuscript already covers

The related-work section already hits the right families: late chunking (Günther et al.), Dense X / RAPTOR / ColBERT--PLAID, Adaptive-RAG / Self-RAG / Adaptive-$k$ / RECOMP / LongLLMLingua, and conformal RAG (TRAQ, CONFLARE, C-RAG). The empirical claim is retrieval-only Pareto structure on SciFact and NFCorpus with MiniLM on CPU.

That package was enough to look competent in 2024. It is thin for 2025--2026 IP&M or a top IR/NLP venue.

## Gap analysis: QALS vs the updated literature

### 1. Index-time context vs query-time assembly

Anthropic contextual retrieval (Sep 2024) and Merola \& Singh (KEIR@ECIR 2025) treat context loss as an indexing problem: prepend LLM context to fixed chunks, or late-encode then pool. Mix-of-Granularity, LGMGC / parent--child chunking (Liu et al., ECIR 2025), LumberChunker (Findings EMNLP 2024), and SmartChunk (ICLR 2026) all choose granularity with a router, logits, an LLM, or a planner. Boundaries still exist before the query, or the query only picks among prebuilt levels.

QALS stores title-prefixed sentence vectors and builds contiguous spans online under budget $B$. That contrast is real. The manuscript understates how crowded the "adaptive granularity" shelf already is. Claiming "what is missing is query-time span assembly" without citing MoG, LGMGC, SmartChunk, and contextual retrieval reads as outdated, not novel.

### 2. Multi-vector SOTA

The paper cites ColBERT / ColBERTv2 / PLAID and then declines to evaluate them. MacAvaney \& Tonellotto (SIGIR 2024) show BM25$\to$ColBERTv2 reranking is a hard efficiency baseline. WARP (SIGIR 2025) and XTR (NeurIPS 2023) are the current multi-vector stack. Sentence multi-vectors are a coarser, cheaper cousin. Without numbers against at least a MiniLM passage baseline plus a lightweight late-interaction or MaxSim-style multi-vector on the same corpus, reviewers will treat the ColBERT paragraph as hand-waving.

### 3. LLM-facing evaluation is absent

The study is retrieval-only. IP&M 2025--2026 RAG papers (AFR-Rank, SC-RAG, MedScaleRE-PF) report LLM listwise reranking, self-corrective generation, or end-task F1. ARES (NAACL 2024) and RAGAS expect context relevance / faithfulness / answer relevance once a generator is in the loop. Adaptive-$k$ is now EMNLP 2025, not an arXiv sketch. Jiang et al. LongRAG and Zhao et al. LongRAG argue about long units and dual-perspective long-context RAG. Corrective RAG (Yan et al.) is a standard robustness baseline.

QALS currently reports nDCG@10 / Recall@10 and prompt whitespace tokens. That is fine as a retrieval paper. It is incomplete as a RAG paper. The EiC note about LLMs is about this gap.

### 4. Conformal story needs a sharper cut

TRAQ calibrates answer sets. CONFLARE calibrates similarity cutoffs. C-RAG certifies generation risk. Conformal-RAG (Feng et al., SIGIR 2025) calibrates sub-claim factuality with group-conditional coverage. QALS calibrates a token budget $B$ from gold-inclusion depth. That object is different. Say so with the 2025 citation in place. Do not imply QALS is the first conformal RAG method.

### 5. Venue bar

Recent IP&M RAG/IR papers run BEIR or TREC-style ranking, LLM rerankers or generators, ablations, and efficiency numbers. A MiniLM-only, two-corpus, retrieval-only Pareto plot sits below that bar unless the framing is explicitly "CPU retrieval / prompt budgeting" and the baselines match that claim.

## Recommended baselines (CPU or small GPU, MiniLM-compatible)

Priority order for a revision that stays runnable on the current stack.

| Baseline | Why | Approx. cost |
|---|---|---|
| Fixed recursive / character windows (already present) | Keep as early-segmentation control | CPU |
| Parent-document / child-match $\to$ parent-return (already present) | Direct foil to query-time packing | CPU |
| Hybrid BM25 + dense RRF (already present) | Strong SciFact ranker in current numbers | CPU |
| **Adaptive-$k$ on the same MiniLM list** | Same encoder; changes $k$, not cuts. Shows QALS is not just variable count | CPU |
| **Title-prefix / cheap contextualization without an LLM** (Anthropic-style context replaced by title or first-$N$ tokens prepended to fixed chunks) | Isolates "context in the vector" from "query-time spans" | CPU |
| **Sentence-level MaxSim** (sum/avg of top sentence similarities without the 1D packer) | Ablates assembly vs multi-vector scoring | CPU |
| **ColBERTv2 as BM25 reranker** on SciFact/NFCorpus (MacAvaney \& Tonellotto recipe) | Honest late-interaction comparison; PLAID full index optional | Small GPU preferred; CPU possible at small $k$ |
| Optional: **RECOMP-extractive** or greedy sentence compression under $B=150$ | Compression after fixed retrieval vs assembly before packing | CPU + small LM if abstractive skipped |
| Optional generation: **tiny instruct model** (e.g. Qwen2.5-1.5B / Llama-3.2-1B) with RAGAS faithfulness + context precision/recall on SciFact claims | Answers the LLM bar without a 70B run | Small GPU |

Skip for a first revision unless compute appears: full PLAID/WARP indexes, Self-RAG fine-tunes, SmartChunk RL planner, Anthropic Claude contextualization at corpus scale.

Minimum table that would change a reviewer's mind:

1. nDCG@10 / Recall@10 (document IDs, $k=10$) for QALS, Adaptive-$k$, parent-doc, hybrid, sentence-MaxSim, BM25$\to$ColBERT-rerank.
2. Mean prompt tokens under a shared budget protocol ($B=150$ or Adaptive-$k$'s selected tokens).
3. One small-LLM faithfulness / context-precision column on SciFact only.

## Honest novelty framing for the next venue

Do not claim QALS invents late interaction, conformal RAG, or adaptive retrieval depth. Those shelves are full.

Claim this instead:

> Dual encoders force a choice between short, specific vectors and long, prompt-heavy contexts. Prior work either contextualizes fixed chunks at index time (late chunking, contextual retrieval), routes among precomputed granularities (MoG, SmartChunk, parent--child chunkers), or adapts how many frozen passages to keep (Adaptive-$k$, Adaptive-RAG). QALS keeps sentence multi-vectors for matching, then solves a small 1D span program at query time under an explicit token budget, and optionally sets that budget by split-conformal calibration of gold-inclusion depth. The contribution is query-time contiguous assembly with a coverage statement on prompt inclusion, evaluated as a retrieval Pareto point against adaptive-$k$ and parent expansion---not a new generative RAG system.

That framing survives contact with 2025--2026 citations. It also matches what the code actually does.

If the next target is IP&M again, add at least one LLM-facing metric and cite AFR-Rank / SC-RAG as the local bar. If the target is SIGIR / ECIR / ACL Findings, lean harder into the multi-vector and Adaptive-$k$ comparisons and keep generation as optional.

## Suggested citation merges into the manuscript

- Replace arXiv-only Adaptive-$k$ with `taguchi2025adaptivek` (EMNLP 2025).
- Add `anthropic2024contextual`, `merola2025reconstructing`, `zhong2025mix`, `liu2025passage`, `duarte2024lumberchunker`, `zhang2026smartchunk` to the chunking subsection.
- Add `macavaney2024plaid`, `scheerer2025warp`, and briefly `lee2023xtr` to late interaction.
- Add `saadfalcon2024ares`, `jiang2024longrag`, `zhao2024longrag`, `yan2024corrective` where evaluation and long-context RAG are discussed.
- Add `feng2025conformalrag` next to TRAQ / CONFLARE.
- Add `xiong2025afrrank` and `li2026scrag` (and optionally `chen2025medscalere`) when stating why IP&M expects LLM-aware evaluation.

## Verification note

Every DOI / anthology / Springer / ACM link in `references_upgrade.bib` was checked against publisher or arXiv pages in this pass. Anthropic is cited as an engineering post, not a journal article. CRAG is cited as arXiv because the ICLR 2025 conference version was withdrawn. SmartChunk is ICLR 2026 (OpenReview); no ACM/Springer DOI yet.
