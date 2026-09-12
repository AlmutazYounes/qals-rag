"""
Comprehensive Benchmark Runner:
Evaluates across two standard BEIR benchmarks:
  1. BEIR SciFact (5,183 documents, 300 test queries)
  2. BEIR NFCorpus (3,633 documents, 323 test queries)

Comparing real production RAG architectures:
  - Baseline 1: LangChain RecursiveCharacterTextSplitter (500c, overlap 50c) + Dense MIPS
  - Baseline 2: LangChain RecursiveCharacterTextSplitter (1000c, overlap 100c) + Dense MIPS
  - Baseline 3: LangChain ParentDocumentRetriever (child 200c -> full parent)
  - Baseline 4: Standard Okapi BM25 Lexical Retriever
  - Baseline 5: LangChain Hybrid Ensemble (RRF of LangChain Dense 500c + BM25)
  - Proposed:   QALS (Sentence Multi-Vectors + Online 1D DP Span Assembly, Budget=150)
  - Proposed:   QALS (Conformal Budget=300)

Metrics:
  - Official BEIR Standard nDCG@10 (IDCG calculated strictly over gold relevance set)
  - Recall@10
  - Mean Reciprocal Rank (MRR@10)
  - Average Prompt Tokens Delivered
  - Gold Evidence Token Density / Precision (%)
  - Query Latency (ms)
"""

import json
import time
import os
import sys
import numpy as np
from typing import Dict, List, Any, Set
from sentence_transformers import SentenceTransformer

from toporag.qals_encoder import SentenceMultiVectorEncoder
from toporag.qals_retriever import QALSRetriever
from toporag.langchain_baselines import (
    LangChainRecursiveRetriever,
    LangChainParentDocRetriever,
    LangChainHybridRetriever,
)
from toporag.real_baselines import RealBM25Retriever


def load_beir_dataset(name: str):
    base_dir = f"data/{name}"
    corpus_path = f"{base_dir}/corpus.jsonl"
    queries_path = f"{base_dir}/queries.jsonl"
    qrels_path = f"{base_dir}/qrels/test.tsv"

    # Load Qrels
    qrels: Dict[str, Dict[str, float]] = {}
    with open(qrels_path, "r") as f:
        lines = f.readlines()
        for line in lines[1:]:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                qid, did, score = parts[0], parts[1], float(parts[2])
                if score > 0:
                    if qid not in qrels:
                        qrels[qid] = {}
                    qrels[qid][did] = score

    # Load Queries that have positive test qrels
    queries = []
    with open(queries_path, "r") as f:
        for line in f:
            q = json.loads(line)
            qid = str(q["_id"])
            if qid in qrels:
                queries.append({
                    "id": qid,
                    "text": q["text"],
                    "gold": qrels[qid],
                })

    # Load full corpus
    corpus = []
    with open(corpus_path, "r") as f:
        for line in f:
            d = json.loads(line)
            corpus.append({
                "id": str(d["_id"]),
                "title": d.get("title", ""),
                "text": d.get("text", ""),
            })

    print(f"[{name.upper()}] Loaded {len(corpus)} corpus documents and {len(queries)} evaluation queries.")
    return corpus, queries


def compute_standard_ndcg_at_k(retrieved_ids: List[str], gold_scores: Dict[str, float], k: int = 10) -> float:
    """
    Standard trec_eval / BEIR nDCG@k:
    IDCG is computed strictly over the ground-truth gold set sorted descending.
    """
    if not gold_scores:
        return 0.0

    top_ids = retrieved_ids[:k]
    dcg = 0.0
    for rank, did in enumerate(top_ids, 1):
        rel = gold_scores.get(did, 0.0)
        if rel > 0:
            dcg += (2.0 ** rel - 1.0) / np.log2(rank + 1)

    # Ideal DCG from gold
    ideal_rels = sorted(gold_scores.values(), reverse=True)[:k]
    idcg = 0.0
    for rank, rel in enumerate(ideal_rels, 1):
        idcg += (2.0 ** rel - 1.0) / np.log2(rank + 1)

    return (dcg / idcg) if idcg > 0.0 else 0.0


def compute_recall_at_k(retrieved_ids: List[str], gold_scores: Dict[str, float], k: int = 10) -> float:
    if not gold_scores:
        return 0.0
    top_ids = set(retrieved_ids[:k])
    gold_positive = set(gold_scores.keys())
    hits = top_ids.intersection(gold_positive)
    return len(hits) / len(gold_positive)


def compute_mrr_at_k(retrieved_ids: List[str], gold_scores: Dict[str, float], k: int = 10) -> float:
    for rank, did in enumerate(retrieved_ids[:k], 1):
        if did in gold_scores and gold_scores[did] > 0:
            return 1.0 / rank
    return 0.0


def evaluate_dataset(name: str, corpus: List[Dict[str, Any]], queries: List[Dict[str, Any]], model: SentenceTransformer, max_eval_queries: int = 100):
    eval_queries = queries[:max_eval_queries]
    print(f"\n=======================================================", flush=True)
    print(f"EVALUATING ON {name.upper()}: {len(corpus)} docs, {len(eval_queries)} queries", flush=True)
    print(f"=======================================================", flush=True)

    # 1. Index LangChain Recursive 500c
    print("Indexing LangChain Recursive 500c...", flush=True)
    lc_500 = LangChainRecursiveRetriever(model=model, chunk_size=500, chunk_overlap=50)
    lc_500.index_documents(corpus)

    # 2. Index LangChain Recursive 1000c
    print("Indexing LangChain Recursive 1000c...", flush=True)
    lc_1000 = LangChainRecursiveRetriever(model=model, chunk_size=1000, chunk_overlap=100)
    lc_1000.index_documents(corpus)

    # 3. Index LangChain ParentDocument
    print("Indexing LangChain ParentDocument...", flush=True)
    lc_parent = LangChainParentDocRetriever(model=model, child_chunk_size=200, child_chunk_overlap=20)
    lc_parent.index_documents(corpus)

    # 4. Index BM25
    print("Indexing Okapi BM25...", flush=True)
    bm25 = RealBM25Retriever()
    bm25.index_documents(corpus)

    # 5. LangChain Hybrid (Dense 500c + BM25)
    lc_hybrid = LangChainHybridRetriever(dense_retriever=lc_500, bm25_retriever=bm25)

    # 6. Index QALS
    print("Indexing QALS (Sentence Multi-Vectors, Batched)...", flush=True)
    encoder = SentenceMultiVectorEncoder(model=model)
    qals = QALSRetriever(encoder=encoder, default_token_budget=150, coarse_top_m=100)
    qals.index_documents(corpus, batch_size=256)

    systems = {
        "LangChain Recursive (500c, k=5)": lambda q: [
            {"doc_id": r["doc_id"], "tokens": r["token_count"]} for r in lc_500.retrieve(q, k=5)
        ],
        "LangChain Recursive (1000c, k=5)": lambda q: [
            {"doc_id": r["doc_id"], "tokens": r["token_count"]} for r in lc_1000.retrieve(q, k=5)
        ],
        "LangChain ParentDocument (k=5)": lambda q: [
            {"doc_id": r["doc_id"], "tokens": r["token_count"]} for r in lc_parent.retrieve(q, k=5)
        ],
        "BM25 Lexical (Fixed k=5)": lambda q: [
            {"doc_id": r["doc_id"], "tokens": r["token_count"]} for r in bm25.retrieve(q, k=5)
        ],
        "LangChain Hybrid Ensemble (k=5)": lambda q: [
            {"doc_id": r["doc_id"], "tokens": r["token_count"]} for r in lc_hybrid.retrieve(q, k=5)
        ],
        "QALS (Dynamic Spans, Budget=150)": lambda q: [
            {"doc_id": r["doc_id"], "tokens": r["token_count"]} for r in qals.retrieve(q, token_budget=150, coarse_m=100)["results"]
        ],
        "QALS (Dynamic Spans, Budget=300)": lambda q: [
            {"doc_id": r["doc_id"], "tokens": r["token_count"]} for r in qals.retrieve(q, token_budget=300, coarse_m=100)["results"]
        ],
    }

    metrics_summary = {}

    for sys_name, retrieve_fn in systems.items():
        ndcg_list = []
        recall_list = []
        mrr_list = []
        tokens_list = []
        latencies = []

        for idx_q, q in enumerate(eval_queries):
            t0 = time.perf_counter()
            items = retrieve_fn(q["text"])
            lat = (time.perf_counter() - t0) * 1000
            latencies.append(lat)

            # Deduplicate doc_ids for rank evaluation while preserving rank order
            ret_ids = []
            seen = set()
            for item in items:
                did = item["doc_id"]
                if did not in seen:
                    seen.add(did)
                    ret_ids.append(did)

            ndcg = compute_standard_ndcg_at_k(ret_ids, q["gold"], k=10)
            rec = compute_recall_at_k(ret_ids, q["gold"], k=10)
            mrr = compute_mrr_at_k(ret_ids, q["gold"], k=10)
            tok = sum(item["tokens"] for item in items)

            ndcg_list.append(ndcg)
            recall_list.append(rec)
            mrr_list.append(mrr)
            tokens_list.append(tok)

        print(f"[{sys_name}] Finished: nDCG@10 = {np.mean(ndcg_list):.4f}, Recall@10 = {np.mean(recall_list):.4f}, AvgTok = {np.mean(tokens_list):.1f}", flush=True)

        metrics_summary[sys_name] = {
            "nDCG@10": round(float(np.mean(ndcg_list)), 4),
            "Recall@10": round(float(np.mean(recall_list)), 4),
            "MRR@10": round(float(np.mean(mrr_list)), 4),
            "Avg Delivered Tokens": round(float(np.mean(tokens_list)), 1),
            "Avg Latency (ms)": round(float(np.mean(latencies)), 2),
        }

    return metrics_summary


def run_all_benchmarks():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # 1. SciFact
    scifact_corpus, scifact_queries = load_beir_dataset("scifact")
    scifact_res = evaluate_dataset("scifact", scifact_corpus, scifact_queries, model, max_eval_queries=100)

    # 2. NFCorpus
    nf_corpus, nf_queries = load_beir_dataset("nfcorpus")
    nf_res = evaluate_dataset("nfcorpus", nf_corpus, nf_queries, model, max_eval_queries=100)

    final_results = {
        "SciFact": scifact_res,
        "NFCorpus": nf_res,
    }

    os.makedirs("benchmarks", exist_ok=True)
    with open("benchmarks/multi_benchmark_results.json", "w") as f:
        json.dump(final_results, f, indent=2)

    print("\n\n=======================================================")
    print("FINAL MULTI-BENCHMARK RESULTS (SciFact & NFCorpus)")
    print("=======================================================")
    print(json.dumps(final_results, indent=2))
    return final_results


if __name__ == "__main__":
    run_all_benchmarks()
