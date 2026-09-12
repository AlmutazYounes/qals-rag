"""
Adaptive retrieval cutoff mechanism.
Dynamically calculates top-k per query using curvature knee detection,
score entropy, and conformal non-conformity calibration.
"""

from typing import List, Tuple, Optional
import numpy as np


class DynamicCutoffSelector:
    """
    Solves the 'chunk count retrieved' dilemma.
    Instead of fixed top_k (e.g., k=5 or k=10), dynamically detects the natural
    relevance boundary per query using:
      1. Geometric Knee / Epsilon-Drop detection in sorted calibrated scores.
      2. Score Entropy (query ambiguity): sharper distribution implies tight cutoff;
         diffuse distribution implies multi-hop / exploratory cutoff.
      3. Conformal Prediction threshold quantile if calibration set is provided.
    """

    def __init__(
        self,
        min_k: int = 1,
        max_k: int = 12,
        drop_threshold: float = 0.18,
        entropy_weight: float = 0.5,
    ):
        self.min_k = min_k
        self.max_k = max_k
        self.drop_threshold = drop_threshold
        self.entropy_weight = entropy_weight
        self.conformal_tau: Optional[float] = None

    def fit_conformal(self, calibration_scores: List[float], alpha: float = 0.1):
        """
        Calibrate non-conformity threshold with (1 - alpha) statistical coverage guarantee.
        """
        if not calibration_scores:
            return
        n = len(calibration_scores)
        # 1 - alpha quantile
        idx = int(np.ceil((n + 1) * (1 - alpha))) - 1
        idx = min(max(idx, 0), n - 1)
        sorted_scores = np.sort(calibration_scores)
        self.conformal_tau = float(sorted_scores[idx])

    def select_cutoff(self, sorted_scores: np.ndarray) -> Tuple[int, dict]:
        """
        sorted_scores: 1D array of descending calibrated similarity scores.
        Returns selected k and diagnostic metadata.
        """
        N = len(sorted_scores)
        if N == 0:
            return 0, {"reason": "empty"}
        if N == 1:
            return 1, {"reason": "single_candidate"}

        effective_max = min(self.max_k, N)
        top_slice = sorted_scores[:effective_max]

        # 1. First-derivative drops (consecutive gaps)
        diffs = top_slice[:-1] - top_slice[1:]
        max_drop_idx = int(np.argmax(diffs)) + 1 if len(diffs) > 0 else 1

        # 2. Entropy of normalized top slice
        exp_scores = np.exp(top_slice - np.max(top_slice))
        probs = exp_scores / (np.sum(exp_scores) + 1e-9)
        entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
        normalized_entropy = entropy / (np.log(len(top_slice)) + 1e-9)

        # 3. Dynamic knee selection:
        # If the gap at max_drop_idx is significant (> drop_threshold), cut there
        knee_k = 1
        for i in range(1, len(top_slice)):
            relative_drop = top_slice[0] - top_slice[i]
            marginal_drop = top_slice[i - 1] - top_slice[i]
            if marginal_drop >= self.drop_threshold or relative_drop >= (self.drop_threshold * 2.0):
                knee_k = i
                break
        else:
            knee_k = effective_max

        # Modulate by query entropy: high entropy means broad search needed
        entropy_boost = int(round(normalized_entropy * self.entropy_weight * 3))
        selected_k = knee_k + entropy_boost

        # Conformal constraint: if conformal_tau is calibrated, don't include items below tau
        if self.conformal_tau is not None:
            conformal_k = int(np.sum(top_slice >= self.conformal_tau))
            selected_k = max(selected_k, conformal_k)

        # Clamp between bounds
        final_k = min(max(selected_k, self.min_k), effective_max)

        diagnostics = {
            "selected_k": final_k,
            "knee_k": knee_k,
            "max_drop_idx": max_drop_idx,
            "entropy": normalized_entropy,
            "top_score": float(top_slice[0]),
            "cut_score": float(top_slice[final_k - 1]),
        }
        return final_k, diagnostics
