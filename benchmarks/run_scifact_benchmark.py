"""
BEIR SciFact Real-World Benchmark Runner.
Compares:
  1. RealFlatDenseRetriever (Fixed chunk size 500 chars, SentenceTransformer all-MiniLM-L6-v2, k=5)
  2. RealParentDocumentRetriever (Child sentence chunks -> Full Parent Document returned, k=5)
  3. RealBM25Retriever (Okapi BM25 lexical baseline, k=5)
  4. QALSRetriever (Query-Adaptive Late Segmentation with sentence multi-vectors and 1D DP span assembly)

Metrics:
  - Hit Rate @ Budget
  - Recall @ Top
  - MRR
  - NDCG
  - Average Delivered Tokens per Query
  - Precision per 100 Delivered Tokens (Token Efficiency)
"""

import json
import time
import os
import numpy as np
from typing import Dict, List, Any

from toporag.qals_encoder import SentenceMultiVectorEncoder
from toporag.qals_retriever import QALSRetriever
from toporag.real_baselines import RealFlatDenseRetriever, RealParentDocumentRetriever, RealBM25Retriever
from toporag.conformal import ConformalBudgetCalibrator


def load_scifact_data(max_docs: int = 500, max_queries: int = 60):
    corpus_path = "data/scifact/corpus.jsonl"
    queries_path = "data/scifact/queries.jsonl"
    qrels_path = "data/scifact/qrels/test.tsv"

    # Load test qrels
    qrels = {}
    with open(qrels_path, "r") as f:
        lines = f.readlines()
        for line in lines[1:]:  # skip header
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                qid, doc_id, score = parts[0], parts[1], float(parts[2])
                if score > 0:
                    if qid not in qrels:
                        qrels[qid] = set()
                    qrels[qid].add(doc_id)

    # Load queries that have positive test qrels
    queries = []
    with open(queries_path, "r") as f:
        for line in f:
            q = json.loads(line)
            qid = str(q["_id"])
            if qid in qrels:
                queries.append({
                    "id": qid,
                    "text": q["text"],
                    "gold_doc_ids": list(qrels[qid]),
                })
            if len(queries) >= max_queries:
                break

    # Get all gold doc IDs needed for these queries to guarantee they exist in corpus
    target_gold_ids = set()
    for q in queries:
        target_gold_ids.update(q["gold_doc_ids"])

    # Load corpus documents: prioritize target gold docs + background distractors
    corpus = []
    background_docs = []

    with open(corpus_path, "r") as f:
        for line in f:
            d = json.loads(line)
            doc_id = str(d["_id"])
            item = {
                "id": doc_id,
                "title": d.get("title", ""),
                "text": d.get("text", ""),
            }
            if doc_id in target_gold_ids:
                corpus.append(item)
            else:
                if len(background_docs) < max_docs:
                    background_docs.append(item)

    # Combine gold docs + background distractors up to max_docs
    corpus.extend(background_docs[: max(0, max_docs - len(corpus))])

    print(f"Loaded {len(corpus)} documents ({len(target_gold_ids)} gold) and {len(queries)} test queries.")
    return corpus, queries


def evaluate_retrievers():
    # 500 documents, 50 queries for fast, statistically solid evaluation on CPU
    corpus, queries = load_scifact_data(max_docs=500, max_queries=50)

    # Split queries into 20 calibration and 30 test
    calib_queries = queries[:15]
    test_queries = queries[15:]
    print(f"Split: {len(calib_queries)} calibration queries, {len(test_queries)} test queries.")

    model_name = "all-MiniLM-L6-v2"
    encoder = SentenceMultiVectorEncoder(model_name=model_name)

    print("\n--- Indexing QALS ---")
    t0 = time.time()
    qals = QALSRetriever(encoder=encoder, default_token_budget=200, coarse_top_m=15)
    qals.index_documents(corpus)
    print(f"QALS indexed in {time.time() - t0:.2f}s")

    print("\n--- Indexing Flat Dense Baseline (Fixed 500 chars) ---")
    t0 = time.time()
    flat_dense = RealFlatDenseRetriever(model_name=model_name, chunk_size=500, overlap=50)
    flat_dense.index_documents(corpus)
    print(f"Flat Dense indexed in {time.time() - t0:.2f}s")

    print("\n--- Indexing Parent-Document Baseline ---")
    t0 = time.time()
    parent_doc = RealParentDocumentRetriever(model_name=model_name)
    parent_doc.index_documents(corpus)
    print(f"Parent-Document indexed in {time.time() - t0:.2f}s")

    print("\n--- Indexing BM25 Baseline ---")
    t0 = time.time()
    bm25 = RealBM25Retriever()
    bm25.index_documents(corpus)
    print(f"BM25 indexed in {time.time() - t0:.2f}s")

    # Calibration step for QALS
    calibrator = ConformalBudgetCalibrator(alpha=0.1)
    calib_budgets = []
    for cq in calib_queries:
        gold_ids = set(cq["gold_doc_ids"])
        # Query with large budget and find rank/tokens needed for first gold hit
        res = qals.retrieve(cq["text"], token_budget=400, coarse_m=20)
        cum_tokens = 0
        found = False
        for item in res["results"]:
            cum_tokens += item["token_count"]
            if item["doc_id"] in gold_ids:
                calib_budgets.append(cum_tokens)
                found = True
                break
        if not found:
            calib_budgets.append(350)

    calibrated_budget = calibrator.calibrate_token_budget(calib_budgets, alpha=0.15)
    print(f"Conformal Calibrated Token Budget (85% coverage guarantee): {calibrated_budget} tokens")

    systems = {
        "BM25 (Fixed k=5)": lambda q: [
            {"doc_id": r["doc_id"], "tokens": r["token_count"], "score": r["score"]}
            for r in bm25.retrieve(q, k=5)
        ],
        "Flat Dense (Fixed k=5)": lambda q: [
            {"doc_id": r["doc_id"], "tokens": r["token_count"], "score": r["score"]}
            for r in flat_dense.retrieve(q, k=5)
        ],
        "Parent-Document (Fixed k=5)": lambda q: [
            {"doc_id": r["doc_id"], "tokens": r["token_count"], "score": r["score"]}
            for r in parent_doc.retrieve(q, k=5)
        ],
        "QALS (Budget=150)": lambda q: [
            {"doc_id": r["doc_id"], "tokens": r["token_count"], "score": r["score"]}
            for r in qals.retrieve(q, token_budget=150)["results"]
        ],
        "QALS (Conformal Budget)": lambda q: [
            {"doc_id": r["doc_id"], "tokens": r["token_count"], "score": r["score"]}
            for r in qals.retrieve(q, token_budget=calibrated_budget)["results"]
        ],
    }

    results = {}
    print("\n--- Evaluating on SciFact Test Set ---")

    for sys_name, retrieve_fn in systems.items():
        hits_count = 0
        mrr_total = 0.0
        ndcg_total = 0.0
        total_tokens = 0
        gold_tokens = 0
        latencies = []

        for q_obj in test_queries:
            q_text = q_obj["text"]
            gold = set(q_obj["gold_doc_ids"])

            t_start = time.perf_counter()
            ret_items = retrieve_fn(q_text)
            lat = (time.perf_counter() - t_start) * 1000
            latencies.append(lat)

            ret_doc_ids = [item["doc_id"] for item in ret_items]
            doc_hits = [1 if did in gold else 0 for did in ret_doc_ids]

            if any(doc_hits):
                hits_count += 1

            # MRR
            rr = 0.0
            for rank, h in enumerate(doc_hits, 1):
                if h:
                    rr = 1.0 / rank
                    break
            mrr_total += rr

            # Standard NDCG with binary relevance
            dcg = sum(h / np.log2(rank + 1) for rank, h in enumerate(doc_hits, 1))
            ideal_hits = sorted(doc_hits, reverse=True)
            idcg = sum(h / np.log2(rank + 1) for rank, h in enumerate(ideal_hits, 1))
            ndcg_total += (dcg / idcg) if idcg > 0 else 0.0

            q_tokens = sum(item["tokens"] for item in ret_items)
            total_tokens += q_tokens
            for item in ret_items:
                if item["doc_id"] in gold:
                    gold_tokens += item["tokens"]

        n_test = len(test_queries)
        avg_tokens = total_tokens / n_test
        token_precision = gold_tokens / max(total_tokens, 1)

        results[sys_name] = {
            "Hit Rate": round(hits_count / n_test, 4),
            "MRR": round(mrr_total / n_test, 4),
            "NDCG": round(ndcg_total / n_test, 4),
            "Avg Delivered Tokens": round(avg_tokens, 1),
            "Signal-to-Noise Ratio (Gold/Total Tokens)": round(token_precision * 100, 2),
            "Avg Latency (ms)": round(float(np.mean(latencies)), 2),
        }

    return results, calibrated_budget


if __name__ == "__main__":
    benchmark_results, cal_b = evaluate_retrievers()
    print("\n=== SCIFACT EMPIRICAL BENCHMARK RESULTS ===")
    print(json.dumps(benchmark_results, indent=2))
    os.makedirs("benchmarks", exist_ok=True)
    with open("benchmarks/scifact_results.json", "w") as f:
        json.dump(benchmark_results, f, indent=2)
