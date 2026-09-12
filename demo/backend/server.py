"""
FastAPI demonstration backend for TopoRAG.
Provides endpoints for indexing documents, comparing TopoRAG vs Standard Flat Dense & BM25,
and returning manifold calibration & dynamic cutoff diagnostics.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import os

from toporag.embeddings import HashFeatureEmbedder
from toporag.retriever import TopoRAGRetriever
from toporag.baselines import FlatDenseRetriever, BM25Retriever
from benchmarks.benchmark import generate_synthetic_evaluation_corpus

app = FastAPI(title="TopoRAG Research Demonstrator", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engines
corpus, sample_queries = generate_synthetic_evaluation_corpus()
embedder = HashFeatureEmbedder(dim=256, anisotropy_strength=0.08)

toporag = TopoRAGRetriever(embedder=embedder)
toporag.index_documents(corpus)

flat_dense = FlatDenseRetriever(embedder=embedder, chunk_size=350, overlap=40)
flat_dense.index_documents(corpus)

bm25 = BM25Retriever(chunk_size=350)
bm25.index_documents(corpus)


class QueryRequest(BaseModel):
    query: str
    toporag_expand: bool = True
    toporag_fixed_k: Optional[int] = None
    flat_k: int = 5


@app.get("/api/corpus")
def get_corpus():
    return {
        "documents": [
            {"id": d["id"], "title": d["title"], "snippet": d["text"][:160] + "..."}
            for d in corpus
        ],
        "sample_queries": [q["query"] for q in sample_queries],
    }


@app.post("/api/retrieve")
def compare_retrieval(req: QueryRequest):
    # TopoRAG retrieval
    topo_res = toporag.retrieve(
        req.query,
        k=req.toporag_fixed_k,
        expand_to_parent=req.toporag_expand,
        return_diagnostics=True,
    )

    # Flat Dense retrieval
    flat_res = flat_dense.retrieve(req.query, k=req.flat_k)

    # BM25 retrieval
    bm25_res = bm25.retrieve(req.query, k=req.flat_k)

    return {
        "query": req.query,
        "toporag": topo_res,
        "flat_dense": {
            "results": flat_res,
            "k": req.flat_k,
        },
        "bm25": {
            "results": bm25_res,
            "k": req.flat_k,
        },
    }


@app.get("/api/benchmark-summary")
def get_benchmark_summary():
    import json
    if os.path.exists("benchmarks/extended_results.json"):
        with open("benchmarks/extended_results.json", "r") as f:
            return json.load(f)
    return {}


# Mount frontend static files
app.mount("/paper", StaticFiles(directory="paper"), name="paper")
app.mount("/", StaticFiles(directory="demo/frontend", html=True), name="static")
