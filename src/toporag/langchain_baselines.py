"""
Production LangChain chunking and retrieval baselines:
1. LangChain RecursiveCharacterTextSplitter (chunk_size=500, chunk_overlap=50) + Dense MIPS
2. LangChain RecursiveCharacterTextSplitter (chunk_size=1000, chunk_overlap=100) + Dense MIPS
3. LangChain ParentDocumentRetriever (Child 200c -> Full Parent Document)
4. LangChain CharacterTextSplitter (Sentence-level / 150c) + Dense MIPS
5. Hybrid Ensemble Retriever (Reciprocal Rank Fusion of LangChain Dense + BM25)
"""

from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter


class LangChainRecursiveRetriever:
    """
    Standard LangChain RAG pipeline:
    RecursiveCharacterTextSplitter + Dense Embedding search (Cosine/Inner Product).
    """

    def __init__(
        self,
        model: SentenceTransformer,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        self.model = model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None

    def index_documents(self, documents: List[Dict[str, Any]], batch_size: int = 128):
        self.chunks = []
        for doc in documents:
            doc_id = str(doc.get("id", doc.get("_id", "")))
            title = doc.get("title", "")
            text = doc.get("text", "")
            content = f"{title}\n\n{text}".strip() if title else text.strip()

            split_texts = self.splitter.split_text(content)
            for idx, st in enumerate(split_texts):
                if st.strip():
                    self.chunks.append({
                        "doc_id": doc_id,
                        "chunk_id": f"{doc_id}_c{idx}",
                        "title": title,
                        "text": st,
                        "token_count": len(st.split()),
                    })

        texts_to_embed = [c["text"] for c in self.chunks]
        self.embeddings = self.model.encode(
            texts_to_embed,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        q_vec = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        sims = self.embeddings @ q_vec
        top_indices = np.argsort(-sims)[:k]
        return [
            {
                "doc_id": self.chunks[idx]["doc_id"],
                "chunk_id": self.chunks[idx]["chunk_id"],
                "text": self.chunks[idx]["text"],
                "score": float(sims[idx]),
                "token_count": self.chunks[idx]["token_count"],
            }
            for idx in top_indices
        ]


class LangChainParentDocRetriever:
    """
    LangChain ParentDocumentRetriever implementation:
    Splits text into small child chunks for high-granularity vector search,
    but retrieves and returns the entire parent document upon matching.
    """

    def __init__(
        self,
        model: SentenceTransformer,
        child_chunk_size: int = 200,
        child_chunk_overlap: int = 20,
    ):
        self.model = model
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
        )
        self.children: List[Dict[str, Any]] = []
        self.parents: Dict[str, Dict[str, Any]] = {}
        self.child_embeddings: Optional[np.ndarray] = None

    def index_documents(self, documents: List[Dict[str, Any]], batch_size: int = 128):
        self.children = []
        self.parents = {}

        for doc in documents:
            doc_id = str(doc.get("id", doc.get("_id", "")))
            title = doc.get("title", "")
            text = doc.get("text", "")
            full = f"{title}\n\n{text}".strip() if title else text.strip()
            self.parents[doc_id] = {
                "doc_id": doc_id,
                "title": title,
                "text": full,
                "token_count": len(full.split()),
            }

            sub_chunks = self.child_splitter.split_text(full)
            for idx, sc in enumerate(sub_chunks):
                if sc.strip():
                    self.children.append({
                        "doc_id": doc_id,
                        "text": sc,
                    })

        child_texts = [c["text"] for c in self.children]
        self.child_embeddings = self.model.encode(
            child_texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
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
                parent = self.parents[doc_id]
                results.append({
                    "doc_id": doc_id,
                    "title": parent["title"],
                    "text": parent["text"],
                    "score": float(sims[idx]),
                    "token_count": parent["token_count"],
                })
                if len(results) >= k:
                    break
        return results


class LangChainHybridRetriever:
    """
    LangChain EnsembleRetriever pattern:
    Combines dense similarity search (LangChain 500c) with Okapi BM25 using
    Reciprocal Rank Fusion (RRF, c=60).
    """

    def __init__(self, dense_retriever: LangChainRecursiveRetriever, bm25_retriever, rrf_c: int = 60):
        self.dense = dense_retriever
        self.bm25 = bm25_retriever
        self.rrf_c = rrf_c

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        dense_hits = self.dense.retrieve(query, k=k * 2)
        bm25_hits = self.bm25.retrieve(query, k=k * 2)

        rrf_scores = {}
        meta_map = {}

        for rank, hit in enumerate(dense_hits):
            did = hit["doc_id"]
            rrf_scores[did] = rrf_scores.get(did, 0.0) + (1.0 / (self.rrf_c + rank + 1))
            meta_map[did] = hit

        for rank, hit in enumerate(bm25_hits):
            did = hit["doc_id"]
            rrf_scores[did] = rrf_scores.get(did, 0.0) + (1.0 / (self.rrf_c + rank + 1))
            if did not in meta_map:
                meta_map[did] = hit

        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: -rrf_scores[x])[:k]
        return [
            {
                "doc_id": did,
                "text": meta_map[did]["text"],
                "score": rrf_scores[did],
                "token_count": meta_map[did]["token_count"],
            }
            for did in sorted_ids
        ]
