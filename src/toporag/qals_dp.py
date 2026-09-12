"""
Exact span assembly under a token budget.

The paper uses at most two non-overlapping contiguous spans of at most ten sentences. For that cardinality the optimum among a shortlist of high-utility intervals is obtained by enumeration.
"""

from typing import List, Dict, Any
import numpy as np


NEG = -1e15


class SpanSegmenterDP:
    def __init__(self, mu_threshold: float = 0.25, kappa_coherence: float = 0.08):
        self.mu_threshold = mu_threshold
        self.kappa_coherence = kappa_coherence

    def find_optimal_spans(
        self,
        sentences: List[Dict[str, Any]],
        sentence_scores: np.ndarray,
        token_budget: int = 150,
        max_spans: int = 2,
    ) -> List[Dict[str, Any]]:
        S = len(sentences)
        if S == 0:
            return []

        scores = np.asarray(sentence_scores, dtype=np.float64).reshape(-1)
        if scores.shape[0] != S:
            scores = np.resize(scores, S)

        tok = [max(int(s.get("token_count", len(str(s.get("text", "")).split()))), 1) for s in sentences]
        B = max(int(token_budget), 1)
        K = max(int(max_spans), 1)
        max_span_len = 10
        candidates = self._enumerate_spans(sentences, scores, tok, B, max_span_len=max_span_len)
        if not candidates:
            best_idx = int(np.argmax(scores))
            return [self._materialize(sentences, scores, best_idx, best_idx, tok[best_idx], float(scores[best_idx]))]

        if K <= 2:
            selected = self._best_k2(candidates, B, K)
        else:
            selected = self._dp_select(candidates, B, K)

        if not selected:
            best_idx = int(np.argmax(scores))
            selected = [self._materialize(sentences, scores, best_idx, best_idx, tok[best_idx], float(scores[best_idx]))]
        return [self._materialize(sentences, scores, c["start_idx"], c["end_idx"], c["token_count"], c["gain"]) for c in selected]

    def _enumerate_spans(self, sentences, scores, tok, B, max_span_len=10):
        S = len(sentences)
        out = []
        for i in range(S):
            acc_t = 0
            acc_g = 0.0
            last = min(S, i + max_span_len)
            for j in range(i, last):
                acc_t += tok[j]
                if acc_t > B:
                    break
                acc_g += scores[j] - self.mu_threshold
                gain = acc_g + self.kappa_coherence * np.log2(j - i + 2)
                out.append({
                    "start_idx": i,
                    "end_idx": j,
                    "token_count": acc_t,
                    "gain": float(gain),
                })
        return out

    def _best_k2(self, candidates, B, K):
        best_gain = NEG
        best = []
        for c in candidates:
            if c["gain"] > best_gain:
                best_gain = c["gain"]
                best = [c]
        if K == 1:
            return best if best_gain > 0 else []

        pool = candidates
        if len(pool) > 40:
            pool = sorted(candidates, key=lambda c: -c["gain"])[:40]
        n = len(pool)
        for a in range(n):
            ca = pool[a]
            for b in range(a + 1, n):
                cb = pool[b]
                if ca["end_idx"] >= cb["start_idx"] and cb["end_idx"] >= ca["start_idx"]:
                    continue
                cost = ca["token_count"] + cb["token_count"]
                if cost > B:
                    continue
                g = ca["gain"] + cb["gain"]
                if g > best_gain:
                    best_gain = g
                    if ca["start_idx"] <= cb["start_idx"]:
                        best = [ca, cb]
                    else:
                        best = [cb, ca]
        return best if best_gain > 0 else []

    def _dp_select(self, candidates, B, K):
        n = len(candidates)
        order = sorted(range(n), key=lambda i: (candidates[i]["end_idx"], candidates[i]["start_idx"]))
        f = np.full((n + 1, B + 1, K + 1), NEG)
        take = np.zeros((n + 1, B + 1, K + 1), dtype=np.int8)
        f[0, 0, 0] = 0.0
        prev_end = []
        for idx, ci in enumerate(order):
            c = candidates[ci]
            p = 0
            for j in range(idx):
                if candidates[order[j]]["end_idx"] < c["start_idx"]:
                    p = j + 1
            prev_end.append(p)
        for i in range(1, n + 1):
            c = candidates[order[i - 1]]
            p = prev_end[i - 1]
            cost = c["token_count"]
            for b in range(B + 1):
                for s in range(K + 1):
                    f[i, b, s] = f[i - 1, b, s]
                    take[i, b, s] = 0
                    if s == 0 or b < cost:
                        continue
                    prev = f[p, b - cost, s - 1]
                    if prev <= NEG / 2:
                        continue
                    val = prev + c["gain"]
                    if val > f[i, b, s]:
                        f[i, b, s] = val
                        take[i, b, s] = 1

        best = NEG
        state = (n, 0, 0)
        for b in range(B + 1):
            for s in range(K + 1):
                if f[n, b, s] > best:
                    best = f[n, b, s]
                    state = (n, b, s)
        if best <= 0:
            return []
        i, b, s = state
        selected = []
        while i > 0 and s >= 0:
            if take[i, b, s] == 1:
                c = candidates[order[i - 1]]
                selected.append(c)
                b -= c["token_count"]
                s -= 1
                i = prev_end[i - 1]
            else:
                i -= 1
        selected.reverse()
        return selected

    def _materialize(self, sentences, scores, i, j, token_count, gain):
        return {
            "start_idx": i,
            "end_idx": j,
            "token_count": int(token_count),
            "gain": float(gain),
            "mean_score": float(np.mean(scores[i : j + 1])),
            "max_score": float(np.max(scores[i : j + 1])),
            "text": " ".join(sentences[k]["text"] for k in range(i, j + 1)),
        }
