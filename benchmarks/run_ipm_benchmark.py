"""
IP&M evaluation runner.

Document ranking uses k=10 unique documents for every system.
Prompt tokens are measured at a production operating point:
  chunk and parent systems concatenate the first 5 retrieved units,
  QALS packs assembled spans under budget B.

QALS ranking score is 0.4 * coarse + 0.6 * max sentence similarity.
Packed tokens are the actual whitespace tokens of selected spans, not a constant.

Split-conformal budgets are fit on BEIR train queries and tested on the official
test split.
"""

import json
import os
import sys
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from toporag.conformal import ConformalBudgetCalibrator
from toporag.qals_dp import SpanSegmenterDP
from toporag.qals_encoder import SentenceMultiVectorEncoder
from toporag.real_baselines import RealBM25Retriever


def load_beir(name: str, qrel_split: str = "test"):
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
    qrel_path = f"{base}/qrels/{qrel_split}.tsv"
    if not os.path.exists(qrel_path):
        return corpus, []
    with open(qrel_path) as f:
        lines = f.readlines()
        start = 1 if lines and lines[0].lower().startswith("query") else 0
        for line in lines[start:]:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                qid, did, score = parts[0], parts[1], float(parts[2])
                if score > 0:
                    qrels.setdefault(qid, {})[did] = score

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
    print(f"[{name.upper()}/{qrel_split}] {len(corpus)} docs, {len(queries)} queries.", flush=True)
    return corpus, queries


def ndcg_at_k(ret_ids: List[str], gold: Dict[str, float], k: int = 10) -> float:
    if not gold:
        return 0.0
    dcg = sum(
        (2.0 ** gold.get(did, 0.0) - 1.0) / np.log2(rank + 1)
        for rank, did in enumerate(ret_ids[:k], 1) if did in gold
    )
    ideal = sorted(gold.values(), reverse=True)[:k]
    idcg = sum((2.0 ** r - 1.0) / np.log2(rank + 1) for rank, r in enumerate(ideal, 1))
    return (dcg / idcg) if idcg > 0.0 else 0.0


def recall_at_k(ret_ids: List[str], gold: Dict[str, float], k: int = 10) -> float:
    if not gold:
        return 0.0
    return len(set(ret_ids[:k]).intersection(gold.keys())) / len(gold)


def unique_docs(hits: List[Dict[str, Any]], limit: int) -> List[str]:
    out = []
    seen = set()
    for h in hits:
        did = h["doc_id"]
        if did not in seen:
            seen.add(did)
            out.append(did)
            if len(out) >= limit:
                break
    return out


def prompt_tokens(hits: List[Dict[str, Any]], n_units: int) -> int:
    return int(sum(h.get("tokens", 0) for h in hits[:n_units]))


def get_or_compute_embeddings(cache_file: str, texts: List[str], model: SentenceTransformer, batch_size: int = 256) -> np.ndarray:
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    if os.path.exists(cache_file):
        print(f"  [CACHE HIT] {cache_file}", flush=True)
        return np.load(cache_file)
    print(f"  [ENCODING] {len(texts)} texts -> {cache_file}", flush=True)
    embs = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
    arr = np.array(embs, dtype=np.float32)
    np.save(cache_file, arr)
    return arr


def pack_qals(candidates: List[Dict[str, Any]], budget: int) -> Tuple[List[str], int]:
    packed_ids = []
    used = 0
    for c in candidates:
        t = int(c["tokens"])
        if packed_ids and used + t > budget:
            continue
        packed_ids.append(c["doc_id"])
        used += t
        if used >= budget:
            break
    return packed_ids, used


def gold_cover_budget(candidates: List[Dict[str, Any]], gold: Dict[str, float], cap: int = 2000) -> int:
    used = 0
    for c in candidates:
        used += int(c["tokens"])
        if c["doc_id"] in gold:
            return used
        if used >= cap:
            break
    return cap


def summarize(name: str, rows: List[Dict[str, float]]) -> Dict[str, float]:
    keys = rows[0].keys()
    out = {k: round(float(np.mean([r[k] for r in rows])), 4 if k != "tokens" else 1) for k in keys}
    if "tokens" in out:
        out["tokens"] = round(float(np.mean([r["tokens"] for r in rows])), 1)
    print(
        f"[{name}] nDCG@10={out['ndcg']:.4f} Recall@10={out['recall']:.4f} tokens={out['tokens']:.1f}",
        flush=True,
    )
    return out


def evaluate_dataset(
    name: str,
    corpus: List[Dict[str, Any]],
    test_queries: List[Dict[str, Any]],
    calib_queries: List[Dict[str, Any]],
    model: SentenceTransformer,
    num_eval: int,
    num_calib: int = 80,
    qals_budget: int = 150,
    coarse_m: int = 100,
    prompt_k: int = 5,
    rank_k: int = 10,
    run_ablations: bool = False,
):
    eval_q = test_queries[:num_eval]
    calib_q = calib_queries[:num_calib]
    cache_dir = f"data/cache_{name}"
    os.makedirs(cache_dir, exist_ok=True)
    print(f"\n=== {name.upper()} test={len(eval_q)} calib={len(calib_q)} ===", flush=True)

    q_texts = [q["text"] for q in eval_q]
    q_vecs = get_or_compute_embeddings(f"{cache_dir}/queries_{num_eval}.npy", q_texts, model, batch_size=64)
    q_tensor = torch.from_numpy(q_vecs)

    print("BM25...", flush=True)
    bm25 = RealBM25Retriever()
    bm25.index_documents(corpus)
    bm25_hits = [bm25.retrieve(q["text"], k=rank_k) for q in eval_q]
    parent_map = {d["id"]: len(f"{d['title']} {d['text']}".split()) for d in corpus}
    for hits in bm25_hits:
        for h in hits:
            h["tokens"] = parent_map.get(h["doc_id"], 0)

    def split_corpus(chunk_size, overlap):
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        chunks = []
        for d in corpus:
            content = f"{d['title']}\n\n{d['text']}".strip() if d["title"] else d["text"].strip()
            for s in splitter.split_text(content):
                if s.strip():
                    chunks.append({"doc_id": d["id"], "text": s, "tokens": len(s.split())})
        return chunks

    def dense_hits(chunks, cache_name):
        emb = get_or_compute_embeddings(f"{cache_dir}/{cache_name}", [c["text"] for c in chunks], model)
        sims = (q_tensor @ torch.from_numpy(emb).T).numpy()
        out = []
        for i in range(len(eval_q)):
            top_idx = np.argsort(-sims[i])[:rank_k]
            out.append([chunks[idx] for idx in top_idx])
        return out, sims

    print("LangChain 500c...", flush=True)
    chunks_500 = split_corpus(500, 50)
    lc500, sims_500 = dense_hits(chunks_500, "emb_500.npy")

    print("LangChain 1000c...", flush=True)
    chunks_1000 = split_corpus(1000, 100)
    lc1000, _ = dense_hits(chunks_1000, "emb_1000.npy")

    print("ParentDocument...", flush=True)
    child = split_corpus(200, 20)
    emb_parent = get_or_compute_embeddings(f"{cache_dir}/emb_parent.npy", [c["text"] for c in child], model)
    sims_parent = (q_tensor @ torch.from_numpy(emb_parent).T).numpy()
    lc_parent = []
    for i in range(len(eval_q)):
        seen = set()
        ret = []
        for idx in np.argsort(-sims_parent[i]):
            did = child[idx]["doc_id"]
            if did in seen:
                continue
            seen.add(did)
            ret.append({"doc_id": did, "tokens": parent_map[did]})
            if len(ret) >= rank_k:
                break
        lc_parent.append(ret)

    print("Hybrid RRF...", flush=True)
    hybrid = []
    for i in range(len(eval_q)):
        rrf = {}
        for rank, h in enumerate(lc500[i]):
            rrf[h["doc_id"]] = rrf.get(h["doc_id"], 0.0) + 1.0 / (60 + rank + 1)
        for rank, h in enumerate(bm25_hits[i]):
            rrf[h["doc_id"]] = rrf.get(h["doc_id"], 0.0) + 1.0 / (60 + rank + 1)
        dids = sorted(rrf, key=lambda x: -rrf[x])[:rank_k]
        hybrid.append([{"doc_id": did, "tokens": parent_map.get(did, 0)} for did in dids])

    print("QALS index...", flush=True)
    encoder = SentenceMultiVectorEncoder(model=model)
    coarse_prompts = []
    all_sentence_prompts = []
    sentence_slices = []
    doc_sentences = []
    for d in corpus:
        title = d.get("title", "")
        text = d.get("text", "")
        coarse_prompts.append((f"{title}. {text}" if title else text)[:600])
        sents = encoder.split_sentences(text)
        if not sents:
            sents = [{
                "text": text if text.strip() else (title or d["id"]),
                "token_count": max(len(text.split()), 1),
            }]
        doc_sentences.append(sents)
        prefix = f"{title[:60]}: " if title else ""
        start = len(all_sentence_prompts)
        for s in sents:
            all_sentence_prompts.append(f"{prefix}{s['text']}")
        sentence_slices.append((start, len(all_sentence_prompts)))

    coarse_vectors = get_or_compute_embeddings(f"{cache_dir}/qals_coarse.npy", coarse_prompts, model)
    sent_vectors = get_or_compute_embeddings(f"{cache_dir}/qals_sent.npy", all_sentence_prompts, model)
    doc_sentence_vectors = [sent_vectors[a:b] for a, b in sentence_slices]
    coarse_sims = (q_tensor @ torch.from_numpy(coarse_vectors).T).numpy()
    segmenter = SpanSegmenterDP()

    def qals_rank(q_v, coarse_row, m=100):
        top = np.argsort(-coarse_row)[:m]
        cands = []
        for didx in top:
            s_vecs = doc_sentence_vectors[didx]
            s_scores = s_vecs @ q_v if len(s_vecs) else np.zeros(1, dtype=np.float32)
            max_s = float(np.max(s_scores)) if len(s_scores) else 0.0
            cands.append({
                "didx": int(didx),
                "doc_id": corpus[didx]["id"],
                "score": 0.4 * float(coarse_row[didx]) + 0.6 * max_s,
                "s_scores": s_scores,
            })
        cands.sort(key=lambda x: -x["score"])
        return cands

    def attach_spans(cands, budget, mu=0.25, kappa=0.08, until_gold=None, max_span_docs=15):
        local = SpanSegmenterDP(mu_threshold=mu, kappa_coherence=kappa)
        seen_gold = False
        for i, c in enumerate(cands):
            if until_gold is None:
                do = i < max_span_docs
            else:
                do = i < 40 and not seen_gold or i < max_span_docs
            if not do:
                c["tokens"] = 10**6
                c["spans"] = []
                continue
            spans = local.find_optimal_spans(
                doc_sentences[c["didx"]], c["s_scores"], token_budget=budget, max_spans=2
            )
            c["spans"] = spans
            c["tokens"] = int(sum(sp["token_count"] for sp in spans))
            if until_gold is not None and c["doc_id"] in until_gold:
                seen_gold = True
        return cands

    print("QALS retrieve...", flush=True)
    qals_cands = []
    for i in range(len(eval_q)):
        ranked = qals_rank(q_vecs[i], coarse_sims[i], m=coarse_m)
        qals_cands.append(attach_spans(ranked, qals_budget, max_span_docs=15))
        if (i + 1) % 10 == 0:
            print(f"  QALS {i+1}/{len(eval_q)}", flush=True)

    def metric_rows(hits_list, n_prompt, packed=False, budget=None):
        rows = []
        for i, q in enumerate(eval_q):
            if packed:
                packed_ids, used = pack_qals(hits_list[i], budget)
                rank_ids = unique_docs(hits_list[i], rank_k)
                rows.append({
                    "ndcg": ndcg_at_k(rank_ids, q["gold"], rank_k),
                    "recall": recall_at_k(rank_ids, q["gold"], rank_k),
                    "ndcg_packed": ndcg_at_k(packed_ids, q["gold"], rank_k),
                    "recall_packed": recall_at_k(packed_ids, q["gold"], rank_k),
                    "tokens": used,
                    "covered": 1.0 if any(d in q["gold"] for d in packed_ids) else 0.0,
                })
            else:
                rank_ids = unique_docs(hits_list[i], rank_k)
                rows.append({
                    "ndcg": ndcg_at_k(rank_ids, q["gold"], rank_k),
                    "recall": recall_at_k(rank_ids, q["gold"], rank_k),
                    "tokens": prompt_tokens(hits_list[i], n_prompt),
                })
        return rows

    report = {
        "LangChain Recursive 500c": summarize(f"{name} LC500", metric_rows(lc500, prompt_k)),
        "LangChain Recursive 1000c": summarize(f"{name} LC1000", metric_rows(lc1000, prompt_k)),
        "LangChain ParentDocument": summarize(f"{name} Parent", metric_rows(lc_parent, prompt_k)),
        "BM25": summarize(f"{name} BM25", metric_rows(bm25_hits, prompt_k)),
        "LangChain Hybrid RRF": summarize(f"{name} Hybrid", metric_rows(hybrid, prompt_k)),
        "QALS B=150": summarize(f"{name} QALS", metric_rows(qals_cands, prompt_k, packed=True, budget=qals_budget)),
    }

    conformal = {}
    if calib_q:
        print("Conformal calibration on train queries...", flush=True)
        calib_texts = [q["text"] for q in calib_q]
        calib_vecs = get_or_compute_embeddings(
            f"{cache_dir}/queries_calib_{len(calib_q)}.npy", calib_texts, model, batch_size=64
        )
        calib_coarse = (torch.from_numpy(calib_vecs) @ torch.from_numpy(coarse_vectors).T).numpy()
        b_stars = []
        for i, q in enumerate(calib_q):
            ranked = qals_rank(calib_vecs[i], calib_coarse[i], m=coarse_m)
            cands = attach_spans(ranked, budget=400, until_gold=q["gold"], max_span_docs=15)
            b_stars.append(gold_cover_budget(cands, q["gold"]))
            if (i + 1) % 20 == 0:
                print(f"  calib {i+1}/{len(calib_q)}", flush=True)
        for alpha in (0.2, 0.1):
            cal = ConformalBudgetCalibrator(alpha=alpha)
            b_hat = cal.calibrate_token_budget(b_stars, alpha=alpha)
            rows = []
            cover = []
            for i, q in enumerate(eval_q):
                packed_ids, used = pack_qals(qals_cands[i], b_hat)
                cover.append(1.0 if any(d in q["gold"] for d in packed_ids) else 0.0)
                rank_ids = unique_docs(qals_cands[i], rank_k)
                rows.append({
                    "ndcg": ndcg_at_k(rank_ids, q["gold"], rank_k),
                    "recall": recall_at_k(rank_ids, q["gold"], rank_k),
                    "ndcg_packed": ndcg_at_k(packed_ids, q["gold"], rank_k),
                    "recall_packed": recall_at_k(packed_ids, q["gold"], rank_k),
                    "tokens": used,
                })
            conformal[f"alpha={alpha}"] = {
                "B_hat": int(b_hat),
                "mean_B_star": round(float(np.mean(b_stars)), 1),
                "coverage": round(float(np.mean(cover)), 4),
                "target": round(1 - alpha, 2),
                "metrics": summarize(f"{name} QALS conformal a={alpha}", rows),
            }
            print(f"  alpha={alpha} B_hat={b_hat} coverage={conformal[f'alpha={alpha}']['coverage']:.3f}", flush=True)
    report["conformal"] = conformal

    ablations = {}
    if run_ablations:
        print("Ablations...", flush=True)
        for label, kwargs in [
            ("B=50", {"budget": 50}),
            ("B=100", {"budget": 100}),
            ("B=200", {"budget": 200}),
            ("kappa=0.00", {"budget": 150, "kappa": 0.0}),
            ("M=20", {"budget": 150, "m": 20}),
        ]:
            budget = kwargs.get("budget", 150)
            mu = kwargs.get("mu", 0.25)
            kappa = kwargs.get("kappa", 0.08)
            m = kwargs.get("m", coarse_m)
            hits = [
                attach_spans(
                    qals_rank(q_vecs[i], coarse_sims[i], m=m),
                    budget, mu=mu, kappa=kappa, max_span_docs=15,
                )
                for i in range(len(eval_q))
            ]
            ablations[label] = summarize(f"{name} {label}", metric_rows(hits, prompt_k, packed=True, budget=budget))
    report["ablations"] = ablations

    out_file = "benchmarks/ipm_benchmark_results.json"
    existing = {}
    if os.path.exists(out_file):
        with open(out_file) as f:
            existing = json.load(f)
    existing[name] = report
    with open(out_file, "w") as f:
        json.dump(existing, f, indent=2)
    return report


def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    for name, n_test, n_calib, ablate in [
        ("scifact", 150, 80, False),
        ("nfcorpus", 100, 80, False),
    ]:
        corpus, test_q = load_beir(name, "test")
        _, train_q = load_beir(name, "train")
        if not train_q:
            train_q = test_q[n_test:]
            print(f"[{name}] no train qrels, using leftover test queries for calibration", flush=True)
        evaluate_dataset(
            name, corpus, test_q, train_q, model,
            num_eval=n_test, num_calib=n_calib, run_ablations=ablate,
        )
    print("\n=== IPM BENCHMARK COMPLETE ===", flush=True)
    with open("benchmarks/ipm_benchmark_results.json") as f:
        print(f.read(), flush=True)


if __name__ == "__main__":
    main()
