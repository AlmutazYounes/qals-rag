"""
End-to-end RAG generation eval for the QALS manuscript upgrade.

Builds prompts from four retrieval systems (whitespace token accounting matches
run_ipm_benchmark.py):
  - Recursive500: top-5 chunks
  - Hybrid RRF: top-5 docs, each truncated to B=150 words
  - ParentDocument: top-5 parents, each truncated to B=150 words
  - QALS: packed spans under total budget B=150

For SciFact: answer SUPPORT / REFUTE / NOT ENOUGH (gold CONTRADICT -> REFUTE).
For NFCorpus: answer the query briefly (no gold EM; report faithfulness only).

Generators (priority order):
  1. OpenAI / Anthropic cheap chat model if OPENAI_API_KEY or ANTHROPIC_API_KEY
     is set in the environment or a local .env file.
  2. HuggingFace instruct model (default Qwen/Qwen2.5-1.5B-Instruct). Skipped
     cleanly if download / load fails.
  3. Always-on local extractive + oracle-answerability proxies (MiniLM, no API).

LLM responses are cached under data/cache_rag_gen/.
Writes benchmarks/rag_generation_results.json.

Optional FiQA / TREC-COVID: pass --datasets if BEIR folders exist under data/.
Skip those downloads when they are slow.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from toporag.qals_dp import SpanSegmenterDP
from toporag.qals_encoder import SentenceMultiVectorEncoder
from toporag.real_baselines import RealBM25Retriever

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "cache_rag_gen"
OUT_PATH = ROOT / "benchmarks" / "rag_generation_results.json"
LABEL_MAP = {"SUPPORT": "SUPPORT", "CONTRADICT": "REFUTE", "REFUTE": "REFUTE"}


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def load_beir(name: str, qrel_split: str = "test"):
    base = ROOT / "data" / name
    corpus = []
    with open(base / "corpus.jsonl") as f:
        for line in f:
            d = json.loads(line)
            corpus.append({
                "id": str(d["_id"]),
                "title": d.get("title", ""),
                "text": d.get("text", ""),
            })

    qrels = {}
    qrel_path = base / "qrels" / f"{qrel_split}.tsv"
    if not qrel_path.exists():
        return corpus, [], {}
    with open(qrel_path) as f:
        lines = f.readlines()
        start = 1 if lines and lines[0].lower().startswith("query") else 0
        for line in lines[start:]:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                qid, did, score = parts[0], parts[1], float(parts[2])
                if score > 0:
                    qrels.setdefault(qid, {})[did] = score

    gold_labels = {}
    queries = []
    with open(base / "queries.jsonl") as f:
        for line in f:
            q = json.loads(line)
            qid = str(q["_id"])
            if qid not in qrels:
                continue
            meta = q.get("metadata") or {}
            label = None
            for _did, evid in meta.items():
                for e in evid:
                    lab = LABEL_MAP.get(str(e.get("label", "")).upper())
                    if lab:
                        label = lab
                        break
                if label:
                    break
            if label:
                gold_labels[qid] = label
            queries.append({
                "id": qid,
                "text": q["text"],
                "gold": qrels[qid],
                "label": label,
            })
    print(f"[{name.upper()}/{qrel_split}] {len(corpus)} docs, {len(queries)} queries, "
          f"{len(gold_labels)} labeled.", flush=True)
    return corpus, queries, gold_labels


def whitespace_tokens(text: str) -> int:
    return len(text.split())


def truncate_words(text: str, budget: int) -> str:
    words = text.split()
    if len(words) <= budget:
        return text.strip()
    return " ".join(words[:budget])


def get_or_compute_embeddings(cache_file: Path, texts: List[str], model: SentenceTransformer, batch_size: int = 256):
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    if cache_file.exists():
        print(f"  [CACHE HIT] {cache_file}", flush=True)
        return np.load(cache_file)
    print(f"  [ENCODING] {len(texts)} texts -> {cache_file}", flush=True)
    embs = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
    arr = np.array(embs, dtype=np.float32)
    np.save(cache_file, arr)
    return arr


def pack_qals(candidates: List[Dict[str, Any]], budget: int) -> Tuple[List[Dict[str, Any]], int]:
    packed = []
    used = 0
    for c in candidates:
        t = int(c["tokens"])
        if packed and used + t > budget:
            continue
        packed.append(c)
        used += t
        if used >= budget:
            break
    return packed, used


def unique_docs(hits: List[Dict[str, Any]], limit: int) -> List[str]:
    out, seen = [], set()
    for h in hits:
        did = h["doc_id"]
        if did not in seen:
            seen.add(did)
            out.append(did)
            if len(out) >= limit:
                break
    return out


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def faithfulness_proxy(answer: str, context: str) -> float:
    """Fraction of answer content words that appear in the context (RAGAS-like)."""
    ans_words = [w.lower() for w in re.findall(r"[a-zA-Z0-9]+", answer)]
    if not ans_words:
        return 0.0
    ctx = set(w.lower() for w in re.findall(r"[a-zA-Z0-9]+", context))
    # Ignore the three SciFact labels themselves.
    stop = {"support", "refute", "not", "enough", "the", "a", "an", "of", "to", "and", "in", "is"}
    content = [w for w in ans_words if w not in stop]
    if not content:
        return 1.0 if answer.strip() else 0.0
    return float(sum(1 for w in content if w in ctx) / len(content))


def parse_scifact_label(text: str) -> str:
    t = text.strip().upper()
    for lab in ("SUPPORT", "REFUTE", "NOT ENOUGH"):
        if lab in t:
            return lab
    if "CONTRADICT" in t:
        return "REFUTE"
    if "INSUFFICIENT" in t or "UNKNOWN" in t:
        return "NOT ENOUGH"
    return "NOT ENOUGH"


def cache_key(system: str, qid: str, backend: str, prompt: str) -> str:
    h = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:16]
    return f"{backend}__{system}__{qid}__{h}.json"


def load_cached(path: Path) -> Optional[Dict[str, Any]]:
    if path.exists():
        return json.loads(path.read_text())
    return None


def save_cached(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


class ChatBackend:
    name = "none"

    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class OpenAIBackend(ChatBackend):
    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model

    def generate(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=64,
        )
        return resp.choices[0].message.content or ""


class AnthropicBackend(ChatBackend):
    name = "anthropic"

    def __init__(self, model: str = "claude-3-5-haiku-latest"):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model

    def generate(self, prompt: str) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=64,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


class HFInstructBackend(ChatBackend):
    name = "hf_instruct"

    def __init__(self, model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"[HF] loading {model_id} ...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        self.model.eval()
        self.model_id = model_id

    def generate(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=48,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()


def resolve_api_backend() -> Optional[ChatBackend]:
    load_dotenv(ROOT / ".env")
    if os.environ.get("OPENAI_API_KEY"):
        try:
            return OpenAIBackend()
        except Exception as e:
            print(f"[WARN] OpenAI backend failed: {e}", flush=True)
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return AnthropicBackend()
        except Exception as e:
            print(f"[WARN] Anthropic backend failed: {e}", flush=True)
    return None


def try_hf_backend(model_id: str) -> Optional[ChatBackend]:
    try:
        return HFInstructBackend(model_id)
    except Exception as e:
        print(f"[WARN] HF instruct path skipped ({model_id}): {e}", flush=True)
        return None


def build_scifact_prompt(claim: str, context: str) -> str:
    return (
        "You are given a scientific claim and retrieved context.\n"
        "Answer with exactly one of: SUPPORT, REFUTE, NOT ENOUGH.\n"
        "Use only the context. If the context is insufficient, answer NOT ENOUGH.\n\n"
        f"Claim: {claim}\n\nContext:\n{context}\n\nAnswer:"
    )


def build_nf_prompt(query: str, context: str) -> str:
    return (
        "Answer the question briefly using only the context. "
        "If the context is insufficient, say NOT ENOUGH.\n\n"
        f"Question: {query}\n\nContext:\n{context}\n\nAnswer:"
    )


def oracle_answer(query: Dict[str, Any], context_doc_ids: List[str]) -> str:
    """Oracle answerability: gold label iff a gold doc appears in the prompt docs."""
    if query.get("label") and any(d in query["gold"] for d in context_doc_ids):
        return query["label"]
    return "NOT ENOUGH"


def extractive_answer(
    claim: str,
    context: str,
    embedder: SentenceTransformer,
    labeled: bool,
) -> str:
    """MiniLM extractive proxy: SUPPORT if a context sentence is close to the claim,
    else NOT ENOUGH. Cannot reliably emit REFUTE without NLI; stays conservative."""
    if not labeled:
        sents = split_sentences(context)[:12]
        if not sents:
            return "NOT ENOUGH"
        qv = embedder.encode([claim], normalize_embeddings=True)
        sv = embedder.encode(sents, normalize_embeddings=True)
        sims = (qv @ np.array(sv).T)[0]
        best = sents[int(np.argmax(sims))]
        return truncate_words(best, 40)

    sents = split_sentences(context)
    if not sents:
        return "NOT ENOUGH"
    qv = embedder.encode([claim], normalize_embeddings=True)[0]
    sv = embedder.encode(sents, normalize_embeddings=True)
    sims = np.array(sv) @ qv
    if float(np.max(sims)) >= 0.45:
        return "SUPPORT"
    return "NOT ENOUGH"


def retrieve_contexts(
    name: str,
    corpus: List[Dict[str, Any]],
    queries: List[Dict[str, Any]],
    model: SentenceTransformer,
    budget: int = 150,
    prompt_k: int = 5,
    rank_k: int = 10,
    coarse_m: int = 100,
) -> Dict[str, List[Dict[str, Any]]]:
    cache_dir = ROOT / "data" / f"cache_{name}"
    eval_q = queries
    q_texts = [q["text"] for q in eval_q]
    q_vecs = get_or_compute_embeddings(
        cache_dir / f"queries_ragen_{len(eval_q)}.npy", q_texts, model, batch_size=64
    )
    # Reuse IPM query cache when sizes match.
    ipm_q = cache_dir / f"queries_{len(eval_q)}.npy"
    if ipm_q.exists() and not (cache_dir / f"queries_ragen_{len(eval_q)}.npy").exists():
        q_vecs = np.load(ipm_q)
    q_tensor = torch.from_numpy(q_vecs)

    parent_map = {d["id"]: f"{d['title']} {d['text']}".strip() for d in corpus}
    parent_tok = {d["id"]: whitespace_tokens(parent_map[d["id"]]) for d in corpus}

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks_500 = []
    for d in corpus:
        content = f"{d['title']}\n\n{d['text']}".strip() if d["title"] else d["text"].strip()
        for s in splitter.split_text(content):
            if s.strip():
                chunks_500.append({"doc_id": d["id"], "text": s, "tokens": whitespace_tokens(s)})

    emb_500 = get_or_compute_embeddings(cache_dir / "emb_500.npy", [c["text"] for c in chunks_500], model)
    sims_500 = (q_tensor @ torch.from_numpy(emb_500).T).numpy()
    lc500 = []
    for i in range(len(eval_q)):
        top_idx = np.argsort(-sims_500[i])[:rank_k]
        lc500.append([chunks_500[idx] for idx in top_idx])

    bm25 = RealBM25Retriever()
    bm25.index_documents(corpus)
    bm25_hits = [bm25.retrieve(q["text"], k=rank_k) for q in eval_q]

    hybrid = []
    for i in range(len(eval_q)):
        rrf = {}
        for rank, h in enumerate(lc500[i]):
            rrf[h["doc_id"]] = rrf.get(h["doc_id"], 0.0) + 1.0 / (60 + rank + 1)
        for rank, h in enumerate(bm25_hits[i]):
            rrf[h["doc_id"]] = rrf.get(h["doc_id"], 0.0) + 1.0 / (60 + rank + 1)
        dids = sorted(rrf, key=lambda x: -rrf[x])[:rank_k]
        hybrid.append([{"doc_id": did, "text": parent_map[did], "tokens": parent_tok[did]} for did in dids])

    child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    child = []
    for d in corpus:
        content = f"{d['title']}\n\n{d['text']}".strip() if d["title"] else d["text"].strip()
        for s in child_splitter.split_text(content):
            if s.strip():
                child.append({"doc_id": d["id"], "text": s})
    emb_parent = get_or_compute_embeddings(cache_dir / "emb_parent.npy", [c["text"] for c in child], model)
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
            ret.append({"doc_id": did, "text": parent_map[did], "tokens": parent_tok[did]})
            if len(ret) >= rank_k:
                break
        lc_parent.append(ret)

    encoder = SentenceMultiVectorEncoder(model=model)
    coarse_prompts, all_sentence_prompts, sentence_slices, doc_sentences = [], [], [], []
    for d in corpus:
        title, text = d.get("title", ""), d.get("text", "")
        coarse_prompts.append((f"{title}. {text}" if title else text)[:600])
        sents = encoder.split_sentences(text)
        if not sents:
            sents = [{"text": text if text.strip() else (title or d["id"]), "token_count": max(whitespace_tokens(text), 1)}]
        doc_sentences.append(sents)
        prefix = f"{title[:60]}: " if title else ""
        start = len(all_sentence_prompts)
        for s in sents:
            all_sentence_prompts.append(f"{prefix}{s['text']}")
        sentence_slices.append((start, len(all_sentence_prompts)))

    coarse_vectors = get_or_compute_embeddings(cache_dir / "qals_coarse.npy", coarse_prompts, model)
    sent_vectors = get_or_compute_embeddings(cache_dir / "qals_sent.npy", all_sentence_prompts, model)
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

    qals_contexts = []
    for i in range(len(eval_q)):
        ranked = qals_rank(q_vecs[i], coarse_sims[i], m=coarse_m)
        for j, c in enumerate(ranked[:15]):
            spans = segmenter.find_optimal_spans(
                doc_sentences[c["didx"]], c["s_scores"], token_budget=budget, max_spans=2
            )
            c["spans"] = spans
            c["tokens"] = int(sum(sp["token_count"] for sp in spans))
            c["text"] = " ".join(sp["text"] for sp in spans)
        packed, used = pack_qals(ranked[:15], budget)
        qals_contexts.append({
            "units": packed,
            "tokens": used,
            "doc_ids": unique_docs(packed, rank_k),
            "context": "\n\n".join(u["text"] for u in packed if u.get("text")),
        })
        if (i + 1) % 10 == 0:
            print(f"  QALS gen-ctx {i+1}/{len(eval_q)}", flush=True)

    def pack_truncated(hits_list, per_unit_budget: Optional[int]):
        out = []
        for hits in hits_list:
            units = []
            for h in hits[:prompt_k]:
                text = h.get("text") or parent_map.get(h["doc_id"], "")
                if per_unit_budget is not None:
                    text = truncate_words(text, per_unit_budget)
                units.append({"doc_id": h["doc_id"], "text": text, "tokens": whitespace_tokens(text)})
            ctx = "\n\n".join(u["text"] for u in units)
            out.append({
                "units": units,
                "tokens": sum(u["tokens"] for u in units),
                "doc_ids": [u["doc_id"] for u in units],
                "context": ctx,
            })
        return out

    return {
        "Recursive500": pack_truncated(lc500, None),
        "HybridRRF": pack_truncated(hybrid, budget),
        "ParentDocument": pack_truncated(lc_parent, budget),
        "QALS": qals_contexts,
    }


def evaluate_backend(
    backend_name: str,
    generate_fn,
    name: str,
    queries: List[Dict[str, Any]],
    contexts: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    systems = {}
    for sys_name, ctx_list in contexts.items():
        rows = []
        labeled_rows = []
        for i, q in enumerate(queries):
            ctx = ctx_list[i]
            if name == "scifact":
                prompt = build_scifact_prompt(q["text"], ctx["context"])
            else:
                prompt = build_nf_prompt(q["text"], ctx["context"])

            key = cache_key(sys_name, q["id"], backend_name, prompt)
            cpath = CACHE_DIR / name / key
            cached = load_cached(cpath)
            if cached:
                answer = cached["answer"]
            else:
                answer = generate_fn(q, ctx, prompt)
                save_cached(cpath, {
                    "answer": answer,
                    "system": sys_name,
                    "qid": q["id"],
                    "backend": backend_name,
                    "prompt_tokens": ctx["tokens"],
                })

            pred = parse_scifact_label(answer) if name == "scifact" else answer.strip()
            faith = faithfulness_proxy(answer, ctx["context"])
            row = {
                "qid": q["id"],
                "pred": pred,
                "gold": q.get("label"),
                "tokens": ctx["tokens"],
                "faithfulness": faith,
                "answer": answer,
            }
            rows.append(row)
            if q.get("label"):
                labeled_rows.append(1.0 if pred == q["label"] else 0.0)

        summary = {
            "n": len(rows),
            "n_labeled": len(labeled_rows),
            "accuracy": round(float(np.mean(labeled_rows)), 4) if labeled_rows else None,
            "mean_prompt_tokens": round(float(np.mean([r["tokens"] for r in rows])), 1),
            "mean_faithfulness": round(float(np.mean([r["faithfulness"] for r in rows])), 4),
        }
        systems[sys_name] = {"summary": summary, "per_query": rows}
        print(
            f"[{name}/{backend_name}/{sys_name}] "
            f"acc={summary['accuracy']} tokens={summary['mean_prompt_tokens']} "
            f"faith={summary['mean_faithfulness']}",
            flush=True,
        )
    return systems


def run_dataset(
    name: str,
    num_eval: int,
    model: SentenceTransformer,
    api_backend: Optional[ChatBackend],
    hf_backend: Optional[ChatBackend],
    budget: int = 150,
) -> Dict[str, Any]:
    corpus, queries, _ = load_beir(name, "test")
    queries = queries[:num_eval]
    print(f"\n=== RAG GEN {name} n={len(queries)} ===", flush=True)
    contexts = retrieve_contexts(name, corpus, queries, model, budget=budget)

    results = {
        "n_queries": len(queries),
        "budget_words": budget,
        "token_accounting": "whitespace split, consistent with run_ipm_benchmark.py",
        "backends": {},
    }

    # Always-on local proxies.
    def oracle_fn(q, ctx, prompt):
        return oracle_answer(q, ctx["doc_ids"])

    def extractive_fn(q, ctx, prompt):
        return extractive_answer(q["text"], ctx["context"], model, labeled=(name == "scifact"))

    results["backends"]["oracle_answerability"] = evaluate_backend(
        "oracle_answerability", oracle_fn, name, queries, contexts
    )
    results["backends"]["extractive_minilm"] = evaluate_backend(
        "extractive_minilm", extractive_fn, name, queries, contexts
    )

    if api_backend is not None:
        def api_fn(q, ctx, prompt, backend=api_backend):
            return backend.generate(prompt)
        results["backends"][api_backend.name] = evaluate_backend(
            api_backend.name, api_fn, name, queries, contexts
        )
        results["api_model"] = getattr(api_backend, "model", api_backend.name)
    else:
        results["api_model"] = None
        results["api_note"] = "No OPENAI_API_KEY / ANTHROPIC_API_KEY in env or .env"

    if hf_backend is not None:
        def hf_fn(q, ctx, prompt, backend=hf_backend):
            return backend.generate(prompt)
        results["backends"][hf_backend.name] = evaluate_backend(
            hf_backend.name, hf_fn, name, queries, contexts
        )
        results["hf_model"] = getattr(hf_backend, "model_id", "hf")
    else:
        results["hf_model"] = None
        results["hf_note"] = "HF instruct path skipped or unavailable"

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["scifact", "nfcorpus"])
    parser.add_argument("--num-eval", type=int, default=30, help="Small slice for generation eval")
    parser.add_argument("--budget", type=int, default=150)
    parser.add_argument("--hf-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--skip-hf", action="store_true")
    parser.add_argument("--skip-api", action="store_true")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    api_backend = None if args.skip_api else resolve_api_backend()
    hf_backend = None
    if not args.skip_hf:
        hf_backend = try_hf_backend(args.hf_model)

    existing = {}
    if OUT_PATH.exists():
        existing = json.loads(OUT_PATH.read_text())

    existing.setdefault("meta", {})
    existing["meta"].update({
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "num_eval_default": args.num_eval,
        "notes": (
            "SciFact accuracy only counted when query metadata has SUPPORT/CONTRADICT. "
            "Faithfulness is a lexical overlap proxy, not full RAGAS. "
            "Optional FiQA/TREC-COVID skipped unless present under data/."
        ),
    })

    for name in args.datasets:
        if not (ROOT / "data" / name).is_dir():
            print(f"[SKIP] data/{name} missing (optional FiQA/TREC-COVID not downloaded).", flush=True)
            continue
        existing[name] = run_dataset(
            name, args.num_eval, model, api_backend, hf_backend, budget=args.budget
        )
        OUT_PATH.write_text(json.dumps(existing, indent=2))
        print(f"[WROTE] {OUT_PATH}", flush=True)

    print("\n=== RAG GENERATION EVAL COMPLETE ===", flush=True)
    print(OUT_PATH.read_text()[:4000], flush=True)


if __name__ == "__main__":
    main()
