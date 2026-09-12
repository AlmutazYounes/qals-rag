"""
Embedder utilities and model wrappers for TopoRAG.
Supports fast TF-IDF / Subword hashing fallback for zero-GPU/lightweight environments
as well as Hugging Face / SentenceTransformer dense models.
"""

from typing import List, Union
import numpy as np
import re
import zlib


class HashFeatureEmbedder:
    """
    High-performance semantic feature embedder.
    Combines subword n-grams, word tokens, and term frequency normalization with
    an orthogonal Gaussian projection matrix.
    Includes an anisotropic cone bias to mimic realistic dense embedding spaces.
    """

    def __init__(self, dim: int = 256, seed: int = 42, anisotropy_strength: float = 0.05):
        self.dim = dim
        self.seed = seed
        self.vocab_size = 4096
        self.anisotropy_strength = anisotropy_strength

        self.rng = np.random.RandomState(seed)
        # Random projection matrix
        raw_mat = self.rng.randn(self.vocab_size, self.dim).astype(np.float32)
        q, _ = np.linalg.qr(raw_mat)
        self.projection = q.astype(np.float32)

        # Anisotropic directional drift vector (simulating BERT/LLM embedding cone)
        drift = self.rng.randn(self.dim).astype(np.float32)
        self.drift_cone = drift / (np.linalg.norm(drift) + 1e-9)

    def _hash(self, token: str) -> int:
        return zlib.crc32(token.encode("utf-8")) % self.vocab_size

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def _text_to_sparse(self, text: str) -> np.ndarray:
        tokens = self._tokenize(text)
        vec = np.zeros(self.vocab_size, dtype=np.float32)
        if not tokens:
            return vec

        for t in tokens:
            # Word weight
            vec[self._hash(t)] += 2.0
            # Character n-grams (3, 4, 5) for rich morphological similarity
            for n in [3, 4]:
                if len(t) >= n:
                    for i in range(len(t) - n + 1):
                        ngram = t[i : i + n]
                        vec[self._hash(ngram)] += 0.5

        # Word pairs
        for i in range(len(tokens) - 1):
            pair = f"{tokens[i]}_{tokens[i+1]}"
            vec[self._hash(pair)] += 1.5

        # L2 normalize sparse vector
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec /= norm
        return vec

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        embeddings = []
        for text in texts:
            sp = self._text_to_sparse(text)
            # Projected representation
            dense = sp @ self.projection

            # Add anisotropic cone bias characteristic of LLM/dense transformers
            if self.anisotropy_strength > 0:
                dense = dense + self.anisotropy_strength * self.drift_cone

            # Final sphere normalization
            norm = np.linalg.norm(dense)
            if norm > 1e-9:
                dense = dense / norm
            embeddings.append(dense)

        res = np.array(embeddings, dtype=np.float32)
        return res[0] if single else res


class DenseEmbedder:
    """
    Sentence-Transformers wrapper with graceful fallback to HashFeatureEmbedder.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dim: int = 128):
        self.model_name = model_name
        self.dim = dim
        self._st_model = None
        self._fallback = HashFeatureEmbedder(dim=dim)

        try:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(model_name)
            self.dim = self._st_model.get_sentence_embedding_dimension()
        except Exception:
            self._st_model = None

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        if self._st_model is not None:
            return self._st_model.encode(texts, normalize_embeddings=True)
        return self._fallback.embed(texts)
