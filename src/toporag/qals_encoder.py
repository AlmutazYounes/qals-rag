"""
Sentence-level contextualized multi-vector encoder.
Batched encoding for high-speed indexing across entire corpora.
"""

from typing import List, Dict, Any, Tuple
import re
import numpy as np
from sentence_transformers import SentenceTransformer


class SentenceMultiVectorEncoder:
    """
    Contextualized Sentence-Level Multi-Vector Encoder.
    - Extracts atomic sentences from documents.
    - Encodes each sentence with its surrounding title/document context in vector batches.
    - Produces a coarse document embedding for candidate generation.
    - Produces fine-grained sentence vectors for query-time segmentation.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu", model: SentenceTransformer = None):
        self.model_name = model_name
        self.model = model or SentenceTransformer(model_name, device=device)
        self.dim = getattr(self.model, "get_embedding_dimension", self.model.get_sentence_embedding_dimension)()

    def split_sentences(self, text: str) -> List[Dict[str, Any]]:
        raw_splits = re.split(r"(?<=[.?!])\s+", text.strip())
        sentences = []
        offset = 0
        for s in raw_splits:
            s_clean = s.strip()
            if not s_clean:
                continue
            char_start = text.find(s_clean, offset)
            if char_start == -1:
                char_start = offset
            char_end = char_start + len(s_clean)
            offset = char_end
            tokens = len(s_clean.split())
            sentences.append({
                "text": s_clean,
                "char_start": char_start,
                "char_end": char_end,
                "token_count": tokens,
            })
        return sentences

    def encode_corpus_batch(self, documents: List[Dict[str, Any]], batch_size: int = 256) -> Dict[str, Any]:
        """
        Encodes an entire corpus in high-throughput flat batches:
        1. Pre-splits all documents into sentence prompt lists with doc pointers.
        2. Encodes all coarse document strings in one batched forward pass.
        3. Encodes all sentence prompts in one batched forward pass.
        4. Re-slices sentences back to their respective parent documents.
        """
        doc_ids = []
        doc_titles = []
        doc_texts = []
        coarse_prompts = []
        doc_sentences = []

        all_sentence_prompts = []
        sentence_slices = []  # (start_idx, end_idx) for each document

        curr_sent_idx = 0
        for d in documents:
            did = str(d.get("id", d.get("_id", "")))
            title = d.get("title", "")
            text = d.get("text", "")
            doc_ids.append(did)
            doc_titles.append(title)
            doc_texts.append(text)

            coarse_text = f"{title}. {text}" if title else text
            coarse_prompts.append(coarse_text[:600])

            sents = self.split_sentences(text)
            if not sents:
                sents = [{
                    "text": text if text.strip() else (title or did),
                    "char_start": 0,
                    "char_end": len(text),
                    "token_count": max(len(text.split()), 1),
                }]
            doc_sentences.append(sents)

            prefix = f"{title[:60]}: " if title else ""
            start = curr_sent_idx
            for s in sents:
                p = f"{prefix}{s['text']}"
                all_sentence_prompts.append(p)
            end = len(all_sentence_prompts)
            sentence_slices.append((start, end))
            curr_sent_idx = end

        # Fast batched encoding of coarse vectors
        coarse_vectors = self.model.encode(
            coarse_prompts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # Fast batched encoding of all sentence vectors across corpus
        all_sent_embeddings = self.model.encode(
            all_sentence_prompts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # Slice back to documents
        doc_sentence_vectors = []
        for start, end in sentence_slices:
            doc_sentence_vectors.append(all_sent_embeddings[start:end])

        return {
            "doc_ids": doc_ids,
            "doc_titles": doc_titles,
            "doc_texts": doc_texts,
            "coarse_vectors": np.array(coarse_vectors, dtype=np.float32),
            "doc_sentence_vectors": doc_sentence_vectors,
            "doc_sentences": doc_sentences,
        }

    def encode_document(self, doc_id: str, title: str, text: str) -> Dict[str, Any]:
        """Single document fallback."""
        res = self.encode_corpus_batch([{"id": doc_id, "title": title, "text": text}], batch_size=32)
        return {
            "doc_id": doc_id,
            "title": title,
            "text": text,
            "doc_vector": res["coarse_vectors"][0],
            "sentence_vectors": res["doc_sentence_vectors"][0],
            "sentences": res["doc_sentences"][0],
        }

    def encode_query(self, query: str) -> np.ndarray:
        vec = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        return np.array(vec, dtype=np.float32)
