"""
Sentence-level contextualized multi-vector encoder.
Encodes documents at the sentence level with contextualized embeddings,
producing both a document-level summary vector for coarse candidate generation
and sentence vectors for query-adaptive late segmentation.
"""

from typing import List, Dict, Any, Tuple
import re
import numpy as np
from sentence_transformers import SentenceTransformer


class SentenceMultiVectorEncoder:
    """
    Contextualized Sentence-Level Multi-Vector Encoder.
    - Extracts atomic sentences from documents.
    - Encodes each sentence with its surrounding paragraph/document context
      (contextualized embedding via title and paragraph awareness).
    - Produces a coarse document embedding for candidate generation.
    - Produces fine-grained sentence vectors for query-time segmentation.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()

    def split_sentences(self, text: str) -> List[Dict[str, Any]]:
        """
        Splits text into sentences with token estimates and character offsets.
        """
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

    def encode_document(self, doc_id: str, title: str, text: str) -> Dict[str, Any]:
        """
        Encodes a single document into:
          - doc_vector: Coarse representation (title + full text summary)
          - sentence_vectors: Matrix (S, D) of contextualized sentence embeddings
          - sentence_metadata: Metadata for each sentence
        """
        sentences = self.split_sentences(text)
        if not sentences:
            sentences = [{
                "text": text if text.strip() else title,
                "char_start": 0,
                "char_end": len(text),
                "token_count": max(len(text.split()), 1),
            }]

        # Contextualized sentence prompts: prepend document title/context
        # This gives each sentence document-level awareness
        context_prompts = []
        for s in sentences:
            if title:
                prompt = f"{title}: {s['text']}"
            else:
                prompt = s["text"]
            context_prompts.append(prompt)

        sentence_embeddings = self.model.encode(
            context_prompts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # Coarse document vector: mean pooling of sentences normalized, or title+text encoding
        full_text_prompt = f"{title}. {text}" if title else text
        doc_vector = self.model.encode(
            [full_text_prompt[:1500]],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        return {
            "doc_id": doc_id,
            "title": title,
            "text": text,
            "doc_vector": np.array(doc_vector, dtype=np.float32),
            "sentence_vectors": np.array(sentence_embeddings, dtype=np.float32),
            "sentences": sentences,
        }

    def encode_query(self, query: str) -> np.ndarray:
        """Encodes query into normalized vector."""
        vec = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        return np.array(vec, dtype=np.float32)
