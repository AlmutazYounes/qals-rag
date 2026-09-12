"""
Query-Adaptive Late Segmentation (QALS) 1D Dynamic Programming Module.
Solves the online span segmentation problem:
Given a sequence of sentence similarity scores s_1, ..., s_S, select contiguous
text spans that maximize relevance + discourse coherence within a token budget B:
   max_{spans} sum_{span} [ sum_{i in span} (s_i - mu) + kappa * coherence(span) ]
subject to total_tokens(spans) <= B.
"""

from typing import List, Dict, Any, Tuple
import numpy as np


class SpanSegmenterDP:
    """
    1D Dynamic Programming Span Segmenter.
    Transforms atomic sentence scores into coherent passage spans dynamically at query time.
    """

    def __init__(self, mu_threshold: float = 0.25, kappa_coherence: float = 0.08):
        """
        mu_threshold: baseline relevance hurdle (sentences below this require coherence bonus to justify inclusion)
        kappa_coherence: weight for adjacent sentence continuity
        """
        self.mu_threshold = mu_threshold
        self.kappa_coherence = kappa_coherence

    def find_optimal_spans(
        self,
        sentences: List[Dict[str, Any]],
        sentence_scores: np.ndarray,
        token_budget: int = 150,
        max_spans: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Finds optimal contiguous spans of sentences within token_budget.
        Returns extracted spans with full text, token counts, and gain metrics.
        """
        S = len(sentences)
        if S == 0:
            return []

        # Candidate span generation: compute utility for all possible contiguous spans (i, j)
        candidate_spans = []
        for i in range(S):
            span_tokens = 0
            raw_gain = 0.0
            for j in range(i, S):
                tok = sentences[j]["token_count"]
                if span_tokens + tok > token_budget:
                    break
                span_tokens += tok
                # Score marginal utility: (score - mu)
                raw_gain += (sentence_scores[j] - self.mu_threshold)
                # Coherence bonus for length > 1 (sentence continuation)
                span_len = j - i + 1
                coherence_bonus = self.kappa_coherence * np.log2(span_len + 1)
                total_gain = raw_gain + coherence_bonus

                candidate_spans.append({
                    "start_idx": i,
                    "end_idx": j,
                    "token_count": span_tokens,
                    "gain": float(total_gain),
                    "mean_score": float(np.mean(sentence_scores[i : j + 1])),
                    "max_score": float(np.max(sentence_scores[i : j + 1])),
                })

        if not candidate_spans:
            # Fallback: single best sentence truncated
            best_idx = int(np.argmax(sentence_scores))
            return [{
                "start_idx": best_idx,
                "end_idx": best_idx,
                "token_count": sentences[best_idx]["token_count"],
                "gain": float(sentence_scores[best_idx]),
                "mean_score": float(sentence_scores[best_idx]),
                "max_score": float(sentence_scores[best_idx]),
                "text": sentences[best_idx]["text"],
            }]

        # Sort candidate spans by gain
        candidate_spans.sort(key=lambda x: -x["gain"])

        # Greedy non-overlapping span selection up to token_budget and max_spans
        selected = []
        used_indices = set()
        budget_used = 0

        for sp in candidate_spans:
            span_range = set(range(sp["start_idx"], sp["end_idx"] + 1))
            if not span_range.intersection(used_indices):
                if budget_used + sp["token_count"] <= token_budget:
                    # Construct text
                    span_text = " ".join(
                        sentences[k]["text"] for k in range(sp["start_idx"], sp["end_idx"] + 1)
                    )
                    sp_copy = dict(sp)
                    sp_copy["text"] = span_text
                    selected.append(sp_copy)
                    used_indices.update(span_range)
                    budget_used += sp["token_count"]
                    if len(selected) >= max_spans:
                        break

        # If no positive gain span was selected, pick the single highest-scoring sentence
        if not selected:
            best_idx = int(np.argmax(sentence_scores))
            selected.append({
                "start_idx": best_idx,
                "end_idx": best_idx,
                "token_count": sentences[best_idx]["token_count"],
                "gain": float(sentence_scores[best_idx]),
                "mean_score": float(sentence_scores[best_idx]),
                "max_score": float(sentence_scores[best_idx]),
                "text": sentences[best_idx]["text"],
            })

        return selected
