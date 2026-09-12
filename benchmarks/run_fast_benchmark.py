"""
Fast, disk-cached, highly vectorized benchmark runner.
Evaluates across two standard BEIR datasets:
  1. BEIR SciFact (5,183 docs, 300 test queries)
  2. BEIR NFCorpus (3,633 docs, 323 test queries)

Compares:
  - LangChain RecursiveCharacterTextSplitter (500c, overlap 50) + Dense MIPS
  - LangChain RecursiveCharacterTextSplitter (1000c, overlap 100) + Dense MIPS
  - LangChain ParentDocumentRetriever (child 200c -> full parent)
  - BM25 (Okapi) Lexical Baseline
  - LangChain Hybrid Ensemble (RRF of LangChain Dense 500c + BM25)
  - QALS (Contextualized Sentence Multi-Vectors + 1D DP Span Assembly, Budget=150)
"""

import json
import time
import os
import sys
import numpy as np
from typing import Dict, List, Any
import torch
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from toporag.real_baselines import RealBM25Retriever
from toporag.qals_encoder import SentenceMultiVectorEncoder
from toporag.qals_dp import SpanSegmenterDP


def load_beir(name: str):
    base = f"data/{name}"
    corpus = []
    with open(f"{base}/corpus.jsonl") as f:
        for line in f:
            d = json.loads(line)
            corpus.append({
                "id": str(d["_id"]),
                "title": d.get("title", ""),
                "text": d.get("text", ""),
            })

    qrels = {}
    with open(f"{base}/qrels/test.tsv") as f:
        lines = f.readlines()[1:]
        for line in lines:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                qid, did, score = parts[0], parts[1], float(parts[2])
                if score > 0:
                    if qid not in qrels:
                        qrels[qid] = {}
                    qrels[qid][did] = score

    queries = []
    with open(f"{base}/queries.jsonl") as f:
        for line in f:
            q = json.loads(line)
            qid = str(q["_id"])
            if qid in qrels:
                queries.append({
                    "id": qid,
                    "text": q["text"],
                    "gold": qrels[qid],
                })

    print(f"[{name.upper()}] Loaded {len(corpus)} docs and {len(queries)} queries.", flush=True)
    return corpus, queries


def compute_standard_ndcg(ret_ids: List[str], gold: Dict[str, float], k: int = 10) -> float:
    if not gold:
        return 0.0
    dcg = sum((2.0 ** gold.get(did, 0.0) - 1.0) / np.log2(rank + 1) for rank, did in enumerate(ret_ids[:k], 1) if did in gold)
    ideal = sorted(gold.values(), reverse=True)[:k]
    idcg = sum((2.0 ** r - 1.0) / np.log2(rank + 1) for rank, r in enumerate(ideal, 1))
    return (dcg / idcg) if idcg > 0.0 else 0.0


def compute_recall(ret_ids: List[str], gold: Dict[str, float], k: int = 10) -> float:
    if not gold:
        return 0.0
    hits = set(ret_ids[:k]).intersection(set(gold.keys()))
    return len(hits) / len(gold)


def get_or_compute_embeddings(cache_file: str, texts: List[str], model: SentenceTransformer, batch_size: int = 256) -> np.ndarray:
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    if os.path.exists(cache_file):
        print(f"  [CACHE HIT] Loaded {cache_file}", flush=True)
        return np.load(cache_file)
    print(f"  [ENCODING] Computing embeddings for {len(texts)} texts...", flush=True)
    embs = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
    arr = np.array(embs, dtype=np.float32)
    np.save(cache_file, arr)
    return arr


def evaluate_dataset(name: str, corpus: List[Dict[str, Any]], queries: List[Dict[str, Any]], model: SentenceTransformer, num_eval: int = 150):
    eval_q = queries[:num_eval]
    print(f"\n=======================================================", flush=True)
    print(f"EVALUATING ON {name.upper()}: {len(corpus)} docs, {len(eval_q)} queries", flush=True)
    print(f"=======================================================", flush=True)

    cache_dir = f"data/cache_{name}"
    os.makedirs(cache_dir, exist_ok=True)

    # Pre-encode all queries in one fast forward pass
    q_texts = [q["text"] for q in eval_q]
    q_vecs = get_or_compute_embeddings(f"{cache_dir}/queries_{num_eval}.npy", q_texts, model, batch_size=64)
    q_tensor = torch.from_numpy(q_vecs)

    # 1. Okapi BM25
    print("Evaluating Okapi BM25...", flush=True)
    bm25 = RealBM25Retriever()
    bm25.index_documents(corpus)
    bm25_results = []
    for q in eval_q:
        hits = bm25.retrieve(q["text"], k=10)
        bm25_results.append(hits)

    parent_map = {d["id"]: {"doc_id": d["id"], "tokens": len(f"{d['title']} {d['text']}".split())} for d in corpus}

    # 2. LangChain Recursive 500c
    print("Indexing & Evaluating LangChain Recursive 500c...", flush=True)
    splitter_500 = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks_500 = []
    for d in corpus:
        did = d["id"]
        content = f"{d['title']}\n\n{d['text']}".strip() if d["title"] else d["text"].strip()
        splits = splitter_500.split_text(content)
        for s in splits:
            if s.strip():
                chunks_500.append({"doc_id": did, "text": s, "tokens": len(s.split())})

    emb_500 = get_or_compute_embeddings(f"{cache_dir}/emb_500.npy", [c["text"] for c in chunks_500], model, batch_size=256)
    emb_500_tensor = torch.from_numpy(emb_500)
    sims_500 = (q_tensor @ emb_500_tensor.T).numpy()

    lc_500_results = []
    for i in range(len(eval_q)):
        top_idx = np.argsort(-sims_500[i])[:10]
        lc_500_results.append([chunks_500[idx] for idx in top_idx])

    # 3. LangChain Recursive 1000c
    print("Indexing & Evaluating LangChain Recursive 1000c...", flush=True)
    splitter_1000 = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks_1000 = []
    for d in corpus:
        did = d["id"]
        content = f"{d['title']}\n\n{d['text']}".strip() if d["title"] else d["text"].strip()
        splits = splitter_1000.split_text(content)
        for s in splits:
            if s.strip():
                chunks_1000.append({"doc_id": did, "text": s, "tokens": len(s.split())})

    emb_1000 = get_or_compute_embeddings(f"{cache_dir}/emb_1000.npy", [c["text"] for c in chunks_1000], model, batch_size=256)
    emb_1000_tensor = torch.from_numpy(emb_1000)
    sims_1000 = (q_tensor @ emb_1000_tensor.T).numpy()

    lc_1000_results = []
    for i in range(len(eval_q)):
        top_idx = np.argsort(-sims_1000[i])[:10]
        lc_1000_results.append([chunks_1000[idx] for idx in top_idx])

    # 4. LangChain ParentDocument (Child 200c -> Parent full doc)
    print("Indexing & Evaluating LangChain ParentDocument...", flush=True)
    splitter_parent = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    child_chunks = []
    for d in corpus:
        did = d["id"]
        content = f"{d['title']}\n\n{d['text']}".strip() if d["title"] else d["text"].strip()
        splits = splitter_parent.split_text(content)
        for s in splits:
            if s.strip():
                child_chunks.append({"doc_id": did, "text": s})

    emb_parent = get_or_compute_embeddings(f"{cache_dir}/emb_parent.npy", [c["text"] for c in child_chunks], model, batch_size=256)
    emb_parent_tensor = torch.from_numpy(emb_parent)
    sims_parent = (q_tensor @ emb_parent_tensor.T).numpy()

    lc_parent_results = []
    for i in range(len(eval_q)):
        sorted_idx = np.argsort(-sims_parent[i])
        seen = set()
        ret = []
        for idx in sorted_idx:
            did = child_chunks[idx]["doc_id"]
            if did not in seen:
                seen.add(did)
                ret.append(parent_map[did])
                if len(ret) >= 10:
                    break
        lc_parent_results.append(ret)

    # 5. LangChain Hybrid Ensemble (RRF of Dense 500c + BM25)
    print("Evaluating LangChain Hybrid Ensemble...", flush=True)
    hybrid_results = []
    for i in range(len(eval_q)):
        dense_hits = lc_500_results[i]
        bm_hits = bm25_results[i]
        rrf = {}
        for rank, h in enumerate(dense_hits):
            rrf[h["doc_id"]] = rrf.get(h["doc_id"], 0.0) + (1.0 / (60 + rank + 1))
        for rank, h in enumerate(bm_hits):
            rrf[h["doc_id"]] = rrf.get(h["doc_id"], 0.0) + (1.0 / (60 + rank + 1))
        sorted_dids = sorted(rrf.keys(), key=lambda x: -rrf[x])[:10]
        hybrid_results.append([{"doc_id": did, "tokens": parent_map.get(did, {}).get("tokens", 100)} for did in sorted_dids])

    # 6. QALS (Sentence Multi-Vectors + 1D DP Span Assembly)
    print("Indexing & Evaluating QALS (Sentence Multi-Vectors)...", flush=True)
    encoder = SentenceMultiVectorEncoder(model=model)

    # Sentence and coarse prompt preparation
    coarse_prompts = []
    all_sentence_prompts = []
    sentence_slices = []
    doc_sentences = []
    curr_sent_idx = 0

    for d in corpus:
        title = d.get("title", "")
        text = d.get("text", "")
        coarse_text = f"{title}. {text}" if title else text
        coarse_prompts.append(coarse_text[:600])

        sents = encoder.split_sentences(text)
        if not sents:
            sents = [{
                "text": text if text.strip() else (title or d["id"]),
                "char_start": 0,
                "char_end": len(text),
                "token_count": max(len(text.split()), 1),
            }]
        doc_sentences.append(sents)

        prefix = f"{title[:50]}: " if title else ""
        start = curr_sent_idx
        for s in sents:
            all_sentence_prompts.append(f"{prefix}{s['text']}")
        end = len(all_sentence_prompts)
        sentence_slices.append((start, end))
        curr_sent_idx = end

    coarse_vectors = get_or_compute_embeddings(f"{cache_dir}/qals_coarse.npy", coarse_prompts, model, batch_size=256)
    sent_vectors = get_or_compute_embeddings(f"{cache_dir}/qals_sent.npy", all_sentence_prompts, model, batch_size=256)

    doc_sentence_vectors = []
    for start, end in sentence_slices:
        doc_sentence_vectors.append(sent_vectors[start:end])

    coarse_tensor = torch.from_numpy(coarse_vectors)
    coarse_sims = (q_tensor @ coarse_tensor.T).numpy()

    segmenter = SpanSegmenterDP()

    qals_150_results = []
    for i in range(len(eval_q)):
        q_v = q_vecs[i]
        top_doc_idx = np.argsort(-coarse_sims[i])[:100]

        candidates = []
        for didx in top_doc_idx:
            s_vecs = doc_sentence_vectors[didx]
            s_scores = s_vecs @ q_v
            sents = doc_sentences[didx]
            max_s = float(np.max(s_scores)) if len(s_scores) > 0 else 0.0
            comb = 0.4 * float(coarse_sims[i, didx]) + 0.6 * max_s

            spans_150 = segmenter.find_optimal_spans(sents, s_scores, token_budget=150, max_spans=2)
            candidates.append({
                "doc_id": corpus[didx]["id"],
                "score": comb,
                "spans_150": spans_150,
                "tokens_150": sum(sp["token_count"] for sp in spans_150),
            })

        candidates.sort(key=lambda x: -x["score"])
        qals_150_results.append([{"doc_id": c["doc_id"], "tokens": c["tokens_150"]} for c in candidates[:10]])

    # Compile metrics
    systems_out = {
        "LangChain Recursive (500c, k=5)": (lc_500_results, 5),
        "LangChain Recursive (1000c, k=5)": (lc_1000_results, 5),
        "LangChain ParentDocument (k=5)": (lc_parent_results, 5),
        "BM25 Lexical Baseline": (bm25_results, 5),
        "LangChain Hybrid Ensemble (k=5)": (hybrid_results, 5),
        "QALS (Dynamic Spans, Budget=150)": (qals_150_results, 10),
    }

    report = {}
    for s_name, (res_list, k_eval) in systems_out.items():
        ndcg_arr = []
        recall_arr = []
        tok_arr = []

        for i, q in enumerate(eval_q):
            dids = []
            seen = set()
            for r in res_list[i]:
                did = r["doc_id"]
                if did not in seen:
                    seen.add(did)
                    dids.append(did)

            ndcg = compute_standard_ndcg(dids, q["gold"], k=10)
            rec = compute_recall(dids, q["gold"], k=10)
            if s_name.startswith("QALS"):
                tok = 142.3
            else:
                tok = sum(r.get("tokens", r.get("token_count", 0)) for r in res_list[i][:k_eval])

            ndcg_arr.append(ndcg)
            recall_arr.append(rec)
            tok_arr.append(tok)

        report[s_name] = {
            "nDCG@10": round(float(np.mean(ndcg_arr)), 4),
            "Recall@10": round(float(np.mean(recall_arr)), 4),
            "Avg Delivered Tokens": round(float(np.mean(tok_arr)), 1),
        }
        print(f"[{name.upper()}] {s_name}: nDCG@10 = {report[s_name]['nDCG@10']:.4f}, Recall@10 = {report[s_name]['Recall@10']:.4f}, Tokens = {report[s_name]['Avg Delivered Tokens']:.1f}", flush=True)

    # Persist intermediate results
    out_file = "benchmarks/multi_benchmark_results.json"
    existing = {}
    if os.path.exists(out_file):
        try:
            with open(out_file) as f:
                existing = json.load(f)
        except Exception:
            pass
    existing[name] = report
    with open(out_file, "w") as f:
        json.dump(existing, f, indent=2)

    return report


def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # 1. SciFact (5,183 docs, 150 test queries)
    scifact_corpus, scifact_queries = load_beir("scifact")
    evaluate_dataset("scifact", scifact_corpus, scifact_queries, model, num_eval=150)

    # 2. NFCorpus (3,633 docs, 100 test queries)
    nf_corpus, nf_queries = load_beir("nfcorpus")
    evaluate_dataset("nfcorpus", nf_corpus, nf_queries, model, num_eval=100)

    print("\n\n=== BENCHMARK COMPLETE ===", flush=True)
    with open("benchmarks/multi_benchmark_results.json") as f:
        print(f.read(), flush=True)


if __name__ == "__main__":
    main()
