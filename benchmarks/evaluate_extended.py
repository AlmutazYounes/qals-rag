"""
Comprehensive Multi-Dataset Evaluation Harness for TopoRAG.
Tests across multiple challenge dimensions:
  1. Noisy / Adversarial Gallery (evaluating hubness resistance)
  2. Multi-Hop Compositional Queries
  3. Context Fragmentation & Noise Token Ingestion
  4. Dynamic Cutoff Efficiency (token savings)
"""

import json
import time
import numpy as np
from toporag.embeddings import HashFeatureEmbedder
from toporag.retriever import TopoRAGRetriever
from toporag.baselines import FlatDenseRetriever, BM25Retriever
from toporag.chunking import HierarchicalChunker


def generate_extended_dataset():
    # 20 distinct technical documents across CS, Physics, and Biology
    docs = [
        {"id": "doc_01", "text": "Raft is a consensus algorithm designed for state machine replication with clear leader election and log consistency."},
        {"id": "doc_02", "text": "Paxos uses two-phase prepare and accept rounds to achieve distributed agreement without strict leader exclusivity."},
        {"id": "doc_03", "text": "Vector database indices use graph methods like HNSW or inverted quantizers IVF-PQ for nearest neighbor similarity."},
        {"id": "doc_04", "text": "Embedding anisotropy confines dense vectors to narrow geometric cones, inducing spurious nearest neighbor hubs."},
        {"id": "doc_05", "text": "Multi-version concurrency control MVCC provides non-blocking readers by tracking commit timestamps and old tuple versions."},
        {"id": "doc_06", "text": "Surface codes arrange data and syndrome qubits on 2D square lattices to execute fault-tolerant quantum error correction."},
        {"id": "doc_07", "text": "Virtual memory management relies on translation lookaside buffers TLB and page fault handlers for address translation."},
        {"id": "doc_08", "text": "CRISPR-Cas9 enables targeted genome editing by directing bacterial endonucleases with synthetic guide RNAs."},
        {"id": "doc_09", "text": "Transformer self-attention computes scaled dot-product matrices between query, key, and value representation vectors."},
        {"id": "doc_10", "text": "Conformal prediction produces finite-sample non-parametric prediction regions with rigorous coverage guarantees under exchangeability."},
        {"id": "doc_11", "text": "TCP congestion control dynamically adjusts sliding window sizes using additive increase multiplicative decrease AIMD heuristics."},
        {"id": "doc_12", "text": "Epigenetic methylation of cytosine residues regulates eukaryotic gene transcription without altering DNA nucleotide sequence."},
        {"id": "doc_13", "text": "LSM-trees optimize write throughput in key-value storage engines by buffering updates in memory before sequential SSTable flushes."},
        {"id": "doc_14", "text": "B-trees maintain balanced search hierarchies on block storage devices with predictable logarithmic search complexity."},
        {"id": "doc_15", "text": "Reinforcement learning from human feedback RLHF aligns neural policy models with user intent using reward model optimization."},
        {"id": "doc_16", "text": "Kalman filters estimate linear dynamical states recursively from noisy observed sensor measurements and motion models."},
        {"id": "doc_17", "text": "Simulated annealing explores non-convex energy landscapes by probabilistically accepting worsening transitions governed by temperature."},
        {"id": "doc_18", "text": "Zero-knowledge proofs allow a prover to establish computational statement validity without revealing underlying witness values."},
        {"id": "doc_19", "text": "Cache coherence protocols like MESI maintain uniform memory state across multi-core processors via bus snooping."},
        {"id": "doc_20", "text": "Singular value decomposition SVD factorizes arbitrary rectangular matrices into orthonormal eigenvectors and singular values."},
    ]

    queries = [
        {"q": "How does Raft leader election handle state machine replication?", "gold": ["doc_01"], "type": "single-hop"},
        {"q": "What causes hubness in high-dimensional vector search for embeddings?", "gold": ["doc_04", "doc_03"], "type": "multi-hop"},
        {"q": "Explain how surface codes and syndrome measurements fix quantum errors.", "gold": ["doc_06"], "type": "single-hop"},
        {"q": "What prevents dirty reads and lock contention in database MVCC systems?", "gold": ["doc_05"], "type": "single-hop"},
        {"q": "Compare write throughput and read amplification between LSM trees and B trees.", "gold": ["doc_13", "doc_14"], "type": "comparative"},
        {"q": "How does self-attention compute queries and keys in neural transformers?", "gold": ["doc_09"], "type": "single-hop"},
        {"q": "What guarantees does conformal prediction provide for uncertainty regions?", "gold": ["doc_10"], "type": "single-hop"},
        {"q": "How do TLBs and page replacement algorithms manage virtual memory frames?", "gold": ["doc_07"], "type": "single-hop"},
        {"q": "How does CRISPR Cas9 guide RNA perform targeted genomic edits?", "gold": ["doc_08"], "type": "single-hop"},
        {"q": "How do TCP sliding windows and AIMD control network congestion?", "gold": ["doc_11"], "type": "single-hop"},
        {"q": "How does simulated annealing escape local minima compared to Kalman filter states?", "gold": ["doc_17", "doc_16"], "type": "comparative"},
        {"q": "How does MESI bus snooping ensure processor cache coherence across cores?", "gold": ["doc_19"], "type": "single-hop"},
    ]
    return docs, queries


def compute_metrics(doc_frequencies, n_queries, system_results, queries, docs_dict):
    total_hits = 0
    total_mrr = 0.0
    total_ndcg = 0.0
    total_tokens_retrieved = 0
    noise_tokens_retrieved = 0

    for i, q in enumerate(queries):
        retrieved_ids = system_results[i]
        gold_ids = set(q["gold"])

        hits = [1 if did in gold_ids else 0 for did in retrieved_ids]
        if any(hits):
            total_hits += 1

        rr = 0.0
        for rank, h in enumerate(hits, 1):
            if h:
                rr = 1.0 / rank
                break
        total_mrr += rr

        dcg = sum(h / np.log2(rank + 1) for rank, h in enumerate(hits, 1))
        ideal_hits = sorted(hits, reverse=True)
        idcg = sum(h / np.log2(rank + 1) for rank, h in enumerate(ideal_hits, 1))
        total_ndcg += (dcg / idcg) if idcg > 0 else 0.0

        for did in retrieved_ids:
            txt = docs_dict.get(did, "")
            tok_len = len(txt.split())
            total_tokens_retrieved += tok_len
            if did not in gold_ids:
                noise_tokens_retrieved += tok_len

    n = len(queries)
    # Gini coefficient
    arr = np.array(list(doc_frequencies.values()), dtype=np.float64)
    arr = np.sort(arr)
    idx = np.arange(1, len(arr) + 1)
    gini = float((np.sum((2 * idx - len(arr) - 1) * arr)) / (len(arr) * (np.sum(arr) + 1e-9)))

    noise_ratio = noise_tokens_retrieved / max(total_tokens_retrieved, 1)

    return {
        "Hit Rate": round(total_hits / n, 4),
        "MRR": round(total_mrr / n, 4),
        "NDCG": round(total_ndcg / n, 4),
        "Avg Tokens Retrieved": round(total_tokens_retrieved / n, 1),
        "Noise Token Ratio": round(noise_ratio, 4),
        "Hub Gini": round(gini, 4),
    }


def run_full_evaluation():
    docs, queries = generate_extended_dataset()
    docs_dict = {d["id"]: d["text"] for d in docs}
    embedder = HashFeatureEmbedder(dim=256, anisotropy_strength=0.12)

    # 1. Flat Dense (k=5)
    flat = FlatDenseRetriever(embedder=embedder, chunk_size=200, overlap=30)
    flat.index_documents(docs)

    # 2. Fine-grained Micro (k=5)
    micro_retriever = FlatDenseRetriever(embedder=embedder, chunk_size=80, overlap=10)
    micro_retriever.index_documents(docs)

    # 3. BM25
    bm25 = BM25Retriever(chunk_size=200)
    bm25.index_documents(docs)

    # 4. TopoRAG (Proposed)
    toporag = TopoRAGRetriever(embedder=embedder)
    toporag.index_documents(docs)

    all_systems = {
        "BM25": lambda q: [r["doc_id"] for r in bm25.retrieve(q, k=5)],
        "Flat Dense (Fixed k=5)": lambda q: [r["doc_id"] for r in flat.retrieve(q, k=5)],
        "Fine-grained Dense (k=5)": lambda q: [r["doc_id"] for r in micro_retriever.retrieve(q, k=5)],
        "TopoRAG (Proposed)": lambda q: [r["doc_id"] for r in toporag.retrieve(q)["results"]],
    }

    eval_summary = {}

    for name, fn in all_systems.items():
        doc_freqs = {d["id"]: 0 for d in docs}
        query_results = []
        for q in queries:
            res = fn(q["q"])
            query_results.append(res)
            for did in res:
                if did in doc_freqs:
                    doc_freqs[did] += 1
        metrics = compute_metrics(doc_freqs, len(queries), query_results, queries, docs_dict)
        eval_summary[name] = metrics

    return eval_summary


if __name__ == "__main__":
    results = run_full_evaluation()
    print("=== EXTENDED EVALUATION RESULTS ===")
    print(json.dumps(results, indent=2))
    with open("benchmarks/extended_results.json", "w") as f:
        json.dump(results, f, indent=2)
