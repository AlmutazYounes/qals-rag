"""
Curvature and hubness calibration module.
Counteracts embedding space anisotropy, distance concentration, and hub artifacts.
"""

from typing import Tuple, Optional
import numpy as np


class ManifoldCalibrator:
    """
    Computes local Riemannian metrics and hubness discount factors.
    1. Computes mean k-NN distance / local density r_k(x) for each indexed chunk.
    2. Computes empirical hubness frequency H(x) (how often x appears in k-NN sets).
    3. Transforms standard inner-product / cosine similarity:
       S_calibrated(q, x) = S_raw(q, x) - lambda_hub * r_k(x) - gamma_hub * log(1 + H(x))
    This penalizes high-density hubs that dominate nearest neighbor lists artificially.
    """

    def __init__(self, k_density: int = 10, lambda_hub: float = 0.35, gamma_hub: float = 0.15):
        self.k_density = k_density
        self.lambda_hub = lambda_hub
        self.gamma_hub = gamma_hub
        self.mean_knn_sim: Optional[np.ndarray] = None
        self.hub_scores: Optional[np.ndarray] = None

    def fit(self, embeddings: np.ndarray) -> "ManifoldCalibrator":
        """
        Fits calibration statistics over the gallery embeddings.
        embeddings: (N, D) normalized float array
        """
        N = embeddings.shape[0]
        if N <= 1:
            self.mean_knn_sim = np.zeros(N, dtype=np.float32)
            self.hub_scores = np.zeros(N, dtype=np.float32)
            return self

        # Compute pairwise cosine similarities
        sim_matrix = embeddings @ embeddings.T  # (N, N)
        # Exclude self-similarity by setting diagonal to -infinity
        np.fill_diagonal(sim_matrix, -np.inf)

        k = min(self.k_density, N - 1)
        # Partition to find top k nearest neighbors for each item
        # top_indices for each row:
        top_indices = np.argpartition(sim_matrix, -k, axis=1)[:, -k:]

        # Mean k-NN similarity (local density measure)
        row_indices = np.arange(N)[:, None]
        top_sims = sim_matrix[row_indices, top_indices]
        self.mean_knn_sim = np.mean(top_sims, axis=1).astype(np.float32)

        # Count in-degree (hubness: how many times each item appears in another's top-k)
        in_degrees = np.zeros(N, dtype=np.float32)
        unique_vals, counts = np.unique(top_indices, return_counts=True)
        in_degrees[unique_vals] = counts.astype(np.float32)

        # Normalize in-degrees by expected average k
        self.hub_scores = (in_degrees / max(k, 1)).astype(np.float32)
        return self

    def calibrate(self, raw_sims: np.ndarray, query_emb: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calibrates raw similarity scores (batch_size, N) or (N,).
        Applies Cross-Scale Local Similarity (CSLS) and Hubness Regularization.
        """
        if self.mean_knn_sim is None or self.hub_scores is None:
            return raw_sims

        # Dynamic Inverted Penalization:
        # S_cal = 2 * S_raw - lambda_hub * r_k(gallery) - gamma_hub * log(1 + H_gallery)
        penalties = self.lambda_hub * self.mean_knn_sim + self.gamma_hub * np.log1p(self.hub_scores)
        
        # If raw_sims is 2D (batch of queries)
        if raw_sims.ndim == 2:
            calibrated = raw_sims - penalties[None, :]
        else:
            calibrated = raw_sims - penalties

        return calibrated
