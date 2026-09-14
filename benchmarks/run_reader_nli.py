"""SciFact NLI reader eval: generation proxy without a paid LLM API."""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder, SentenceTransformer

from toporag.qals_dp import SpanSegmenterDP
from toporag.qals_encoder import SentenceMultiVectorEncoder
from toporag.real_baselines import RealBM25Retriever


def load_beir(name: str, split: str = "test"):
    base = f"data/{name}"
    corpus = []
    with open(f"{base}/corpus.jsonl") as f:
        for line in f:
            d = json.loads(line)
            corpus.append({"id": str(d["_id"]), "title": d.get("title", ""), "text": d.get("text", "")})
    qrels: Dict[str, Dict[str, float]] = {}
    with open(f"{base}/qrels/{split}.tsv") as f:
        lines = f.readlines()
        start = 1 if lines and lines[0].lower().startswith("query") else 0
        for line in lines[start:]:
            p = line.strip().split("\t")
            if len(p) >= 3 and float(p[2]) > 0:
                qrels.setdefault(p[0], {})[p[1]] = float(p[2])
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
                    "metadata": q.get("metadata", {}),
                })
    return corpus, queries


def get_emb(path: str, texts: List[str], model: SentenceTransformer) -> np.ndarray:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        return np.load(path)
    arr = np.asarray(
        model.encode(texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False),
        dtype=np.float32,
    )
    np.save(path, arr)
    return arr


def adaptive_k(scores: List[float], max_k: int = 5) -> int:
    if len(scores) < 2:
        return 1
    gaps = [(scores[i] - scores[i + 1]) / (abs(scores[i]) + 1e-8) for i in range(min(len(scores) - 1, max_k))]
    return max(1, min(int(np.argmax(gaps)) + 1, max_k))


def claim_label(meta: Any) -> Optional[str]:
    if not isinstance(meta, dict) or not meta:
        return None
    labels = []
    for v in meta.values():
        if isinstance(v, list):
            for item in v:
                if isinstance(item, dict) and "label" in item:
                    labels.append(str(item["label"]).upper())
    if any(x.startswith("CONTRADICT") or x == "REFUTE" for x in labels):
        return "CONTRADICT"
    if any(x.startswith("SUPPORT") for x in labels):
        return "SUPPORT"
    return None


def ctx_from_units(units: List[Dict[str, Any]], n: Optional[int] = None, budget: Optional[int] = None) -> str:
    parts = []
    used = 0
    for i, u in enumerate(units):
        if n is not None and i >= n:
            break
        text = u.get("text") or ""
        toks = text.split()
        if budget is not None and used + len(toks) > budget and parts:
            remain = budget - used
            if remain > 0:
                parts.append(" ".join(toks[:remain]))
            break
        parts.append(text)
        used += len(toks)
        if budget is not None and used >= budget:
            break
    return "\n\n".join(p for p in parts if p.strip())


def main(num_eval: int = 80, budget: int = 150):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    corpus, queries = load_beir("scifact", "test")
    eval_q = queries[:num_eval]
    cache = "data/cache_scifact"
    os.makedirs(cache, exist_ok=True)

    q_vecs = get_emb(f"{cache}/reader_q_{num_eval}.npy", [q["text"] for q in eval_q], model)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = []
    for d in corpus:
        full = f"{d['title']}\n\n{d['text']}".strip() if d["title"] else d["text"]
        for s in splitter.split_text(full):
            if s.strip():
                chunks.append({"doc_id": d["id"], "text": s, "tokens": len(s.split())})
    c_emb = get_emb(f"{cache}/emb_500.npy", [c["text"] for c in chunks], model)
    sims = q_vecs @ c_emb.T

    bm25 = RealBM25Retriever()
    bm25.index_documents(corpus)
    parent_text = {
        d["id"]: (f"{d['title']}\n\n{d['text']}".strip() if d["title"] else d["text"]) for d in corpus
    }

    encoder = SentenceMultiVectorEncoder(model=model)
    all_prompts: List[str] = []
    slices = []
    doc_sents = []
    coarse_prompts = []
    for d in corpus:
        title, text = d["title"], d["text"]
        coarse_prompts.append((f"{title}. {text}" if title else text)[:600])
        sents = encoder.split_sentences(text)
        if not sents:
            sents = [{"text": text or title or d["id"], "token_count": max(len(text.split()), 1)}]
        doc_sents.append(sents)
        start = len(all_prompts)
        prefix = f"{title[:60]}: " if title else ""
        for s in sents:
            all_prompts.append(f"{prefix}{s['text']}")
        slices.append((start, len(all_prompts)))
    coarse = get_emb(f"{cache}/qals_coarse.npy", coarse_prompts, model)
    sent = get_emb(f"{cache}/qals_sent.npy", all_prompts, model)
    doc_sent_vecs = [sent[a:b] for a, b in slices]
    coarse_sims = q_vecs @ coarse.T
    segmenter = SpanSegmenterDP()

    def qals_pack(i: int) -> List[Dict[str, Any]]:
        top = np.argsort(-coarse_sims[i])[:100]
        cands = []
        for di in top:
            s_scores = doc_sent_vecs[di] @ q_vecs[i] if len(doc_sent_vecs[di]) else np.zeros(1)
            spans = segmenter.find_optimal_spans(doc_sents[di], s_scores, token_budget=budget, max_spans=2)
            pieces = [sp.get("text") or "" for sp in spans]
            text = " ".join(pieces)
            tok = int(sum(int(sp.get("token_count", len((sp.get("text") or "").split()))) for sp in spans))
            cands.append({
                "doc_id": corpus[di]["id"],
                "text": text,
                "tokens": max(tok, 1),
                "score": 0.4 * float(coarse_sims[i, di]) + 0.6 * float(np.max(s_scores) if len(s_scores) else 0),
            })
        cands.sort(key=lambda x: -x["score"])
        packed, used = [], 0
        for c in cands:
            if packed and used + c["tokens"] > budget:
                continue
            packed.append(c)
            used += c["tokens"]
            if used >= budget:
                break
        return packed

    systems: Dict[str, List[str]] = {
        "Recursive500_k5": [],
        "AdaptiveK_Recursive500": [],
        "Hybrid_docs_B150": [],
        "QALS_B150": [],
    }
    gold_labels: List[Optional[str]] = []
    labeled_idx: List[int] = []
    for i, q in enumerate(eval_q):
        top = np.argsort(-sims[i])[:10]
        hits = [chunks[j] for j in top]
        scores = [float(sims[i, j]) for j in top]
        k = adaptive_k(scores, max_k=5)
        systems["Recursive500_k5"].append(ctx_from_units(hits, n=5))
        systems["AdaptiveK_Recursive500"].append(ctx_from_units(hits, n=k))

        bm = bm25.retrieve(q["text"], k=10)
        hybrid = []
        for h in bm:
            did = h["doc_id"]
            hybrid.append({
                "doc_id": did,
                "text": parent_text.get(did, ""),
                "tokens": len(parent_text.get(did, "").split()),
            })
        seen = {h["doc_id"] for h in hybrid}
        for h in hits:
            if h["doc_id"] not in seen:
                hybrid.insert(0, {
                    "doc_id": h["doc_id"],
                    "text": parent_text.get(h["doc_id"], ""),
                    "tokens": len(parent_text.get(h["doc_id"], "").split()),
                })
                seen.add(h["doc_id"])
        systems["Hybrid_docs_B150"].append(ctx_from_units(hybrid, budget=budget))
        systems["QALS_B150"].append(ctx_from_units(qals_pack(i), budget=budget))

        lab = claim_label(q.get("metadata"))
        gold_labels.append(lab)
        if lab is not None:
            labeled_idx.append(i)

    print(f"Labeled claims: {len(labeled_idx)}/{num_eval}", flush=True)
    print("Loading NLI cross-encoder/nli-deberta-v3-xsmall ...", flush=True)
    nli = CrossEncoder("cross-encoder/nli-deberta-v3-xsmall")

    results = {
        "num_eval": num_eval,
        "budget": budget,
        "n_labeled": len(labeled_idx),
        "model": "cross-encoder/nli-deberta-v3-xsmall",
        "systems": {},
    }
    for name, contexts in systems.items():
        pairs = [[contexts[i][:3500], eval_q[i]["text"]] for i in range(num_eval)]
        scores = nli.predict(pairs, batch_size=8)
        preds = []
        for row in scores:
            idx = int(np.argmax(row))
            preds.append(["CONTRADICT", "SUPPORT", "NEUTRAL"][idx])
        tok = [len(c.split()) for c in contexts]
        correct = sum(1 for i in labeled_idx if preds[i] == gold_labels[i])
        acc = correct / max(1, len(labeled_idx))
        inclusion = []
        for i, q in enumerate(eval_q):
            ctx = contexts[i].lower()
            ok = False
            for did in q["gold"]:
                snippet = parent_text.get(did, "")[:120].lower()
                if snippet and snippet[:40] in ctx:
                    ok = True
                    break
            inclusion.append(1.0 if ok else 0.0)
        results["systems"][name] = {
            "label_accuracy": round(acc, 4),
            "n_labeled": len(labeled_idx),
            "neutral_rate": round(float(np.mean([p == "NEUTRAL" for p in preds])), 4),
            "mean_prompt_tokens": round(float(np.mean(tok)), 1),
            "gold_snippet_inclusion": round(float(np.mean(inclusion)), 4),
            "note": "NLI reader proxy for generation. No paid LLM API key on this machine.",
        }
        print(name, results["systems"][name], flush=True)

    path = "benchmarks/reader_nli_results.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print("Wrote", path, flush=True)


if __name__ == "__main__":
    main()
