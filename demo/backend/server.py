"""
FastAPI demonstration backend for QALS (Query-Adaptive Late Segmentation).
Provides interactive endpoints comparing:
  - QALS (Contextualized Sentence Multi-Vectors + 1D DP Dynamic Span Assembly)
  - Standard Flat Dense (Fixed 500-char chunks, k=5)
  - Parent-Document Retriever (Child sentence -> Full Document returned)
  - BM25 Lexical (Okapi BM25 on full document)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import os
import json

from toporag.qals_encoder import SentenceMultiVectorEncoder
from toporag.qals_retriever import QALSRetriever
from toporag.real_baselines import RealFlatDenseRetriever, RealParentDocumentRetriever, RealBM25Retriever
from benchmarks.run_scifact_benchmark import load_scifact_data

app = FastAPI(title="QALS Research Demonstrator", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load real SciFact subset (120 docs, 20 queries for instant responsiveness)
corpus, queries = load_scifact_data(max_docs=120, max_queries=20)
model_name = "all-MiniLM-L6-v2"
encoder = SentenceMultiVectorEncoder(model_name=model_name)

print("Indexing demo models...")
qals = QALSRetriever(encoder=encoder, default_token_budget=150, coarse_top_m=12)
qals.index_documents(corpus)

flat_dense = RealFlatDenseRetriever(model_name=model_name, chunk_size=500, overlap=50)
flat_dense.index_documents(corpus)

parent_doc = RealParentDocumentRetriever(model_name=model_name)
parent_doc.index_documents(corpus)

bm25 = RealBM25Retriever()
bm25.index_documents(corpus)
print("Demo models ready.")


class QueryRequest(BaseModel):
    query: str
    qals_token_budget: int = 150
    flat_k: int = 5


@app.get("/api/corpus")
def get_corpus():
    return {
        "documents": [
            {"id": d["id"], "title": d["title"], "snippet": d["text"][:160] + "..."}
            for d in corpus[:10]
        ],
        "sample_queries": [q["text"] for q in queries[:6]],
    }


@app.post("/api/retrieve")
def compare_retrieval(req: QueryRequest):
    # 1. QALS retrieval
    qals_res = qals.retrieve(req.query, token_budget=req.qals_token_budget, return_diagnostics=True)

    # 2. Flat Dense retrieval
    flat_res = flat_dense.retrieve(req.query, k=req.flat_k)

    # 3. Parent-Document retrieval
    parent_res = parent_doc.retrieve(req.query, k=req.flat_k)

    # 4. BM25 retrieval
    bm25_res = bm25.retrieve(req.query, k=req.flat_k)

    return {
        "query": req.query,
        "qals": qals_res,
        "flat_dense": {
            "results": flat_res,
            "total_tokens": sum(r["token_count"] for r in flat_res),
            "k": req.flat_k,
        },
        "parent_doc": {
            "results": parent_res,
            "total_tokens": sum(r["token_count"] for r in parent_res),
            "k": req.flat_k,
        },
        "bm25": {
            "results": bm25_res,
            "total_tokens": sum(r["token_count"] for r in bm25_res),
            "k": req.flat_k,
        },
    }


@app.get("/api/benchmark-summary")
def get_benchmark_summary():
    if os.path.exists("benchmarks/scifact_results.json"):
        with open("benchmarks/scifact_results.json", "r") as f:
            return json.load(f)
    return {}


# Mount static files
app.mount("/paper", StaticFiles(directory="paper"), name="paper")
app.mount("/", StaticFiles(directory="demo/frontend", html=True), name="static")
