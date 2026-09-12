"""
Split-conformal prediction for retrieval token budgeting.
Calibrates token budget B or score threshold tau on held-out calibration queries
to statistically guarantee (1 - alpha) empirical coverage:
    P(Gold document or evidence span in retrieved context) >= 1 - alpha.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np


class ConformalBudgetCalibrator:
    """
    Split-Conformal Predictor for RAG Retrieval.
    Given calibration queries with known relevant documents, determines the minimal
    score threshold or token budget required to cover the gold evidence with 1 - alpha confidence.
    """

    def __init__(self, alpha: float = 0.1):
        """
        alpha: target error rate (e.g. 0.1 for 90% confidence guarantee)
        """
        self.alpha = alpha
        self.calibrated_threshold: float = 0.0
        self.calibrated_budget: int = 250
        self.is_calibrated: bool = False

    def calibrate_from_scores(
        self,
        calibration_records: List[Dict[str, Any]],
        alpha: Optional[float] = None,
    ) -> float:
        """
        calibration_records: list of dicts, each with:
          - 'gold_doc_scores': list of cosine/retrieval scores of gold documents for this query
        Computes non-conformity scores:
          E_i = 1.0 - max(scores of gold docs for query i)
        The conformal threshold tau is determined such that:
          P(S_gold >= tau) >= 1 - alpha
        """
        if alpha is not None:
            self.alpha = alpha

        n = len(calibration_records)
        if n == 0:
            self.calibrated_threshold = 0.3
            self.is_calibrated = True
            return self.calibrated_threshold

        gold_max_scores = []
        for rec in calibration_records:
            scores = rec.get("gold_doc_scores", [])
            if scores:
                gold_max_scores.append(float(np.max(scores)))
            else:
                gold_max_scores.append(0.0)

        # Non-conformity scores: higher means less conforming (worse score for gold doc)
        non_conformity = 1.0 - np.array(gold_max_scores)

        # Conformal quantile: ceil((n + 1) * (1 - alpha)) / n
        q_idx = int(np.ceil((n + 1) * (1 - self.alpha))) - 1
        q_idx = min(max(q_idx, 0), n - 1)

        sorted_nc = np.sort(non_conformity)
        nc_threshold = sorted_nc[q_idx]

        self.calibrated_threshold = float(1.0 - nc_threshold)
        self.is_calibrated = True
        return self.calibrated_threshold

    def calibrate_token_budget(
        self,
        required_token_budgets: List[int],
        alpha: Optional[float] = None,
    ) -> int:
        """
        Calibrates global token budget B:
        Given for each calibration query the minimum tokens needed to retrieve the gold evidence,
        compute the (1 - alpha) empirical upper quantile.
        """
        if alpha is not None:
            self.alpha = alpha

        n = len(required_token_budgets)
        if n == 0:
            self.calibrated_budget = 300
            self.is_calibrated = True
            return self.calibrated_budget

        q_idx = int(np.ceil((n + 1) * (1 - self.alpha))) - 1
        q_idx = min(max(q_idx, 0), n - 1)

        sorted_budgets = np.sort(required_token_budgets)
        self.calibrated_budget = int(sorted_budgets[q_idx])
        self.is_calibrated = True
        return self.calibrated_budget
