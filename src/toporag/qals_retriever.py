"""
Query-Adaptive Late Segmentation (QALS) Retrieval Engine.
Integrates:
  1. Contextualized Sentence Multi-Vector Representation
  2. Coarse Document-Level Candidate Filtering
  3. Fine-Grained Sentence Scoring with Late Interaction
  4. 1D Dynamic Programming Span Segmentation for Adaptive Context Assembly
  5. Split-Conformal Budget Calibration
"""

from typing import List, Dict, Any, Optional
import numpy as np

from toporag.qals_encoder import SentenceMultiVectorEncoder
from toporag.qals_dp import SpanSegmenterDP
from toporag.conformal import ConformalBudgetCalibrator


class QALSRetriever:
    """
    Query-Adaptive Late Segmentation (QALS) Engine.
    Discards fixed index-time chunk boundaries.
    Assembles contiguous evidence spans dynamically at query time based on sentence relevance
    and discourse coherence constraints.
    """

    def __init__(
        self,
        encoder: Optional[SentenceMultiVectorEncoder] = None,
        segmenter: Optional[SpanSegmenterDP] = None,
        calibrator: Optional[ConformalBudgetCalibrator] = None,
        default_token_budget: int = 250,
        coarse_top_m: int = 15,
    ):
        self.encoder = encoder or SentenceMultiVectorEncoder()
        self.segmenter = segmenter or SpanSegmenterDP()
        self.calibrator = calibrator or ConformalBudgetCalibrator(alpha=0.1)
        self.default_token_budget = default_token_budget
        self.coarse_top_m = coarse_top_m

        # Corpus storage
        self.doc_ids: List[str] = []
        self.doc_titles: List[str] = []
        self.doc_texts: List[str] = []
        self.coarse_vectors: Optional[np.ndarray] = None
        self.doc_sentence_vectors: List[np.ndarray] = []
        self.doc_sentences: List[List[Dict[str, Any]]] = []
        self.is_indexed: bool = False

    def index_documents(self, documents: List[Dict[str, Any]], show_progress: bool = False, batch_size: int = 256):
        """
        Indexes documents via high-speed batched encoding.
        """
        enc_res = self.encoder.encode_corpus_batch(documents, batch_size=batch_size)
        self.doc_ids = enc_res["doc_ids"]
        self.doc_titles = enc_res["doc_titles"]
        self.doc_texts = enc_res["doc_texts"]
        self.coarse_vectors = enc_res["coarse_vectors"]
        self.doc_sentence_vectors = enc_res["doc_sentence_vectors"]
        self.doc_sentences = enc_res["doc_sentences"]
        self.is_indexed = True

    def retrieve(
        self,
        query: str,
        token_budget: Optional[int] = None,
        coarse_m: Optional[int] = None,
        return_diagnostics: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes query-adaptive retrieval:
        1. Encodes query.
        2. Coarse filtering: top M candidate documents.
        3. Fine sentence scoring: compute dot products with sentence multi-vectors.
        4. Dynamic span segmentation: extract optimal coherent spans within token_budget.
        """
        if not self.is_indexed or self.coarse_vectors is None:
            raise ValueError("QALS index is empty. Call index_documents() first.")

        budget = token_budget or self.default_token_budget
        m = coarse_m or self.coarse_top_m
        m = min(m, len(self.doc_ids))

        # 1. Query vector
        q_vec = self.encoder.encode_query(query)

        # 2. Coarse candidate ranking (dot product of normalized vectors)
        coarse_scores = self.coarse_vectors @ q_vec
        candidate_indices = np.argsort(-coarse_scores)[:m]

        # 3. Fine-grained sentence scoring across top candidates
        doc_candidates = []
        for doc_idx in candidate_indices:
            sent_vecs = self.doc_sentence_vectors[doc_idx]
            sent_scores = sent_vecs @ q_vec
            sentences = self.doc_sentences[doc_idx]

            # Dynamic span segmentation for this document
            spans = self.segmenter.find_optimal_spans(
                sentences,
                sent_scores,
                token_budget=min(budget, 180),
                max_spans=2,
            )

            max_sent_score = float(np.max(sent_scores)) if len(sent_scores) > 0 else 0.0
            mean_sent_score = float(np.mean(sent_scores)) if len(sent_scores) > 0 else 0.0

            # Document score is a blend of coarse score and top sentence peak (MaxSim style)
            combined_score = 0.4 * float(coarse_scores[doc_idx]) + 0.6 * max_sent_score

            doc_candidates.append({
                "doc_id": self.doc_ids[doc_idx],
                "title": self.doc_titles[doc_idx],
                "coarse_score": float(coarse_scores[doc_idx]),
                "max_sentence_score": max_sent_score,
                "combined_score": combined_score,
                "spans": spans,
            })

        # Rank candidates by combined score
        doc_candidates.sort(key=lambda x: -x["combined_score"])

        # 4. Global token budget packing: pack spans from highest-ranked documents
        delivered_results = []
        total_tokens_used = 0

        for cand in doc_candidates:
            cand_tokens = sum(s["token_count"] for s in cand["spans"])
            if total_tokens_used + cand_tokens <= budget or len(delivered_results) == 0:
                delivered_results.append({
                    "doc_id": cand["doc_id"],
                    "title": cand["title"],
                    "score": round(cand["combined_score"], 4),
                    "coarse_score": round(cand["coarse_score"], 4),
                    "max_sentence_score": round(cand["max_sentence_score"], 4),
                    "spans": [s["text"] for s in cand["spans"]],
                    "assembled_context": " ... ".join(s["text"] for s in cand["spans"]),
                    "token_count": cand_tokens,
                })
                total_tokens_used += cand_tokens
                if total_tokens_used >= budget:
                    break

        output = {
            "query": query,
            "results": delivered_results,
            "total_tokens_delivered": total_tokens_used,
            "token_budget": budget,
            "candidate_pool_size": m,
        }

        if return_diagnostics:
            output["diagnostics"] = {
                "top_doc_score": delivered_results[0]["score"] if delivered_results else 0.0,
                "num_documents_packed": len(delivered_results),
                "budget_utilization": round(total_tokens_used / max(budget, 1), 3),
            }

        return output
