"""
Real-world baselines for public benchmark evaluation:
1. FlatDenseRetriever: Fixed chunking (e.g. 512 chars) with single-vector embedding and top-k
2. ParentDocumentRetriever: Small child chunks indexed, parent returned on hit
3. BM25Retriever: Lexical Okapi BM25 on fixed chunks
"""

from typing import List, Dict, Any, Optional
import numpy as np
import re
from sentence_transformers import SentenceTransformer


class RealFlatDenseRetriever:
    """Standard fixed-chunk dense retrieval using real SentenceTransformer embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 500, overlap: int = 50, device: str = "cpu"):
        self.model = SentenceTransformer(model_name, device=device)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None

    def index_documents(self, documents: List[Dict[str, Any]], show_progress: bool = False):
        self.chunks = []
        for doc in documents:
            doc_id = str(doc.get("id", doc.get("_id", "")))
            title = doc.get("title", "")
            text = doc.get("text", "")
            full = f"{title}. {text}" if title else text
            step = max(self.chunk_size - self.overlap, self.chunk_size // 2)

            for i in range(0, len(full), step):
                chunk_str = full[i : i + self.chunk_size].strip()
                if chunk_str:
                    self.chunks.append({
                        "id": f"{doc_id}_c{i}",
                        "doc_id": doc_id,
                        "title": title,
                        "text": chunk_str,
                        "token_count": len(chunk_str.split()),
                    })

        texts = [c["text"] for c in self.chunks]
        self.embeddings = self.model.encode(
            texts, batch_size=64, normalize_embeddings=True, show_progress_bar=show_progress
        )

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        q_vec = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        sims = self.embeddings @ q_vec
        top_indices = np.argsort(-sims)[:k]
        return [
            {
                "doc_id": self.chunks[idx]["doc_id"],
                "title": self.chunks[idx]["title"],
                "text": self.chunks[idx]["text"],
                "score": float(sims[idx]),
                "token_count": self.chunks[idx]["token_count"],
            }
            for idx in top_indices
        ]


class RealParentDocumentRetriever:
    """
    Parent-Document retriever:
    Indexes small child chunks (e.g. 150 chars / 30 tokens),
    but when a child is retrieved, resolves to the entire document or larger parent.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", child_size: int = 150, device: str = "cpu"):
        self.model = SentenceTransformer(model_name, device=device)
        self.child_size = child_size
        self.children: List[Dict[str, Any]] = []
        self.doc_map: Dict[str, Dict[str, Any]] = {}
        self.child_embeddings: Optional[np.ndarray] = None

    def index_documents(self, documents: List[Dict[str, Any]], show_progress: bool = False):
        self.children = []
        self.doc_map = {}

        for doc in documents:
            doc_id = str(doc.get("id", doc.get("_id", "")))
            title = doc.get("title", "")
            text = doc.get("text", "")
            self.doc_map[doc_id] = {
                "id": doc_id,
                "title": title,
                "text": text,
                "token_count": len(text.split()),
            }

            # Child splits (sentence or sub-paragraph)
            splits = re.split(r"(?<=[.?!])\s+", text.strip())
            for s_idx, s in enumerate(splits):
                s_clean = s.strip()
                if len(s_clean) > 10:
                    child_prompt = f"{title}: {s_clean}" if title else s_clean
                    self.children.append({
                        "child_id": f"{doc_id}_s{s_idx}",
                        "doc_id": doc_id,
                        "text": s_clean,
                        "prompt": child_prompt,
                    })

        prompts = [c["prompt"] for c in self.children]
        self.child_embeddings = self.model.encode(
            prompts, batch_size=64, normalize_embeddings=True, show_progress_bar=show_progress
        )

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        q_vec = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        sims = self.child_embeddings @ q_vec
        top_indices = np.argsort(-sims)

        results = []
        seen_parents = set()

        for idx in top_indices:
            doc_id = self.children[idx]["doc_id"]
            if doc_id not in seen_parents:
                seen_parents.add(doc_id)
                parent_doc = self.doc_map[doc_id]
                results.append({
                    "doc_id": doc_id,
                    "title": parent_doc["title"],
                    "text": parent_doc["text"],
                    "matched_child": self.children[idx]["text"],
                    "score": float(sims[idx]),
                    "token_count": parent_doc["token_count"],
                })
                if len(results) >= k:
                    break
        return results


class RealBM25Retriever:
    """Okapi BM25 implementation on full document texts."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: List[Dict[str, Any]] = []
        self.doc_lens: List[int] = []
        self.avg_doc_len: float = 0.0
        self.df: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_term_freqs: List[Dict[str, int]] = []

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def index_documents(self, documents: List[Dict[str, Any]]):
        self.docs = []
        self.doc_lens = []
        self.doc_term_freqs = []
        self.df = {}

        for doc in documents:
            doc_id = str(doc.get("id", doc.get("_id", "")))
            title = doc.get("title", "")
            text = doc.get("text", "")
            full = f"{title} {text}".strip()
            tokens = self._tokenize(full)
            self.docs.append({"doc_id": doc_id, "title": title, "text": text, "token_count": len(tokens)})
            self.doc_lens.append(len(tokens))

            tf: Dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self.doc_term_freqs.append(tf)

            for t in set(tokens):
                self.df[t] = self.df.get(t, 0) + 1

        N = len(self.docs)
        self.avg_doc_len = float(np.mean(self.doc_lens)) if self.doc_lens else 1.0

        for term, freq in self.df.items():
            self.idf[term] = float(np.log((N - freq + 0.5) / (freq + 0.5) + 1.0))

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        q_tokens = self._tokenize(query)
        scores = np.zeros(len(self.docs), dtype=np.float32)

        for i, tf in enumerate(self.doc_term_freqs):
            dl = self.doc_lens[i]
            score = 0.0
            for t in q_tokens:
                if t in tf:
                    freq = tf[t]
                    num = freq * (self.k1 + 1)
                    denom = freq + self.k1 * (1 - self.b + self.b * (dl / self.avg_doc_len))
                    score += self.idf.get(t, 0.0) * (num / denom)
            scores[i] = score

        top_indices = np.argsort(-scores)[:k]
        return [
            {
                "doc_id": self.docs[idx]["doc_id"],
                "title": self.docs[idx]["title"],
                "text": self.docs[idx]["text"],
                "score": float(scores[idx]),
                "token_count": self.docs[idx]["token_count"],
            }
            for idx in top_indices
        ]
