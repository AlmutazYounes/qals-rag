"""
Baseline retrievers for comparative evaluation:
1. Standard Flat Dense Retriever (Fixed chunk size, fixed top-k, raw cosine similarity)
2. Small Chunk Fixed Retriever
3. BM25 / Lexical Retriever
4. Hierarchical Parent-Document Retriever (without manifold calibration or dynamic cutoff)
"""

from typing import List, Dict, Any, Optional
import numpy as np
import re


class FlatDenseRetriever:
    """Standard dense similarity search with fixed chunk size and fixed top-k."""

    def __init__(self, embedder, chunk_size: int = 500, overlap: int = 50):
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None

    def index_documents(self, documents: List[Dict[str, str]]):
        self.chunks = []
        for doc in documents:
            doc_id = doc.get("id", "")
            text = doc.get("text", "")
            step = max(self.chunk_size - self.overlap, self.chunk_size // 2)
            for i in range(0, len(text), step):
                chunk_str = text[i : i + self.chunk_size]
                if chunk_str.strip():
                    self.chunks.append({
                        "id": f"{doc_id}_chunk_{i}",
                        "doc_id": doc_id,
                        "text": chunk_str,
                    })

        texts = [c["text"] for c in self.chunks]
        self.embeddings = self.embedder.embed(texts)

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        query_emb = self.embedder.embed(query)
        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)
        sims = (query_emb @ self.embeddings.T)[0]
        top_k_indices = np.argsort(-sims)[:k]
        results = []
        for idx in top_k_indices:
            results.append({
                "id": self.chunks[idx]["id"],
                "doc_id": self.chunks[idx]["doc_id"],
                "text": self.chunks[idx]["text"],
                "score": float(sims[idx]),
            })
        return results


class BM25Retriever:
    """Lexical Okapi BM25 retriever baseline."""

    def __init__(self, k1: float = 1.5, b: float = 0.75, chunk_size: int = 500):
        self.k1 = k1
        self.b = b
        self.chunk_size = chunk_size
        self.chunks: List[Dict[str, Any]] = []
        self.doc_len: List[int] = []
        self.avg_doc_len: float = 0.0
        self.df: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_term_freqs: List[Dict[str, int]] = []

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def index_documents(self, documents: List[Dict[str, str]]):
        self.chunks = []
        for doc in documents:
            doc_id = doc.get("id", "")
            text = doc.get("text", "")
            step = max(self.chunk_size - 50, self.chunk_size // 2)
            for i in range(0, len(text), step):
                c_text = text[i : i + self.chunk_size]
                if c_text.strip():
                    self.chunks.append({"id": f"{doc_id}_{i}", "doc_id": doc_id, "text": c_text})

        N = len(self.chunks)
        self.doc_len = []
        self.doc_term_freqs = []
        self.df = {}

        for c in self.chunks:
            tokens = self._tokenize(c["text"])
            self.doc_len.append(len(tokens))
            tf: Dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self.doc_term_freqs.append(tf)
            for t in set(tokens):
                self.df[t] = self.df.get(t, 0) + 1

        self.avg_doc_len = float(np.mean(self.doc_len)) if self.doc_len else 1.0

        for term, freq in self.df.items():
            self.idf[term] = float(np.log((N - freq + 0.5) / (freq + 0.5) + 1.0))

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        q_tokens = self._tokenize(query)
        scores = np.zeros(len(self.chunks), dtype=np.float32)

        for i, tf in enumerate(self.doc_term_freqs):
            dl = self.doc_len[i]
            score = 0.0
            for t in q_tokens:
                if t in tf:
                    freq = tf[t]
                    numerator = freq * (self.k1 + 1)
                    denominator = freq + self.k1 * (1 - self.b + self.b * (dl / self.avg_doc_len))
                    score += self.idf.get(t, 0.0) * (numerator / denominator)
            scores[i] = score

        top_indices = np.argsort(-scores)[:k]
        return [
            {
                "id": self.chunks[idx]["id"],
                "doc_id": self.chunks[idx]["doc_id"],
                "text": self.chunks[idx]["text"],
                "score": float(scores[idx]),
            }
            for idx in top_indices
        ]
