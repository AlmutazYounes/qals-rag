"""
Unit tests for QALS (Query-Adaptive Late Segmentation):
- SentenceMultiVectorEncoder
- SpanSegmenterDP
- ConformalBudgetCalibrator
- QALSRetriever end-to-end
"""

import unittest
import numpy as np
from toporag.qals_encoder import SentenceMultiVectorEncoder
from toporag.qals_dp import SpanSegmenterDP
from toporag.conformal import ConformalBudgetCalibrator
from toporag.qals_retriever import QALSRetriever


class TestQALS(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.encoder = SentenceMultiVectorEncoder(model_name="all-MiniLM-L6-v2")

    def test_sentence_splitting_and_encoding(self):
        text = (
            "Diffusion tensor imaging evaluates cerebral white matter development in newborns. "
            "Microstructural changes reflect myelination and axonal organization during gestation. "
            "Premature birth often disrupts these developmental pathways."
        )
        enc = self.encoder.encode_document("doc1", "Cerebral Development", text)
        self.assertEqual(enc["doc_id"], "doc1")
        self.assertEqual(len(enc["sentences"]), 3)
        self.assertEqual(enc["sentence_vectors"].shape[0], 3)
        self.assertEqual(enc["sentence_vectors"].shape[1], 384)
        self.assertEqual(enc["doc_vector"].shape, (384,))

    def test_span_segmenter_dp(self):
        segmenter = SpanSegmenterDP(mu_threshold=0.2, kappa_coherence=0.08)
        sentences = [
            {"text": "Sentence A is background context.", "token_count": 5},
            {"text": "Sentence B contains the key answer to the question.", "token_count": 9},
            {"text": "Sentence C elaborates on the exact findings.", "token_count": 7},
            {"text": "Sentence D is an unrelated topic.", "token_count": 6},
        ]
        # B and C have high scores
        scores = np.array([0.15, 0.75, 0.70, 0.10], dtype=np.float32)

        spans = segmenter.find_optimal_spans(sentences, scores, token_budget=20, max_spans=1)
        self.assertEqual(len(spans), 1)
        # Should select contiguous span containing B and C
        self.assertIn("Sentence B", spans[0]["text"])
        self.assertIn("Sentence C", spans[0]["text"])

    def test_conformal_calibrator(self):
        calibrator = ConformalBudgetCalibrator(alpha=0.1)
        # Sample required token budgets across 10 calibration queries
        budgets = [80, 95, 120, 140, 150, 160, 175, 190, 210, 240]
        calibrated_b = calibrator.calibrate_token_budget(budgets, alpha=0.1)
        self.assertGreaterEqual(calibrated_b, 190)

    def test_qals_retriever_end_to_end(self):
        retriever = QALSRetriever(encoder=self.encoder, default_token_budget=100)
        docs = [
            {
                "id": "med_01",
                "title": "Cerebral White Matter",
                "text": "Diffusion tensor MRI reveals microstructural white matter development in newborn infants. Axonal fibers align during third trimester.",
            },
            {
                "id": "astro_01",
                "title": "Black Hole Mergers",
                "text": "Gravitational wave detectors LIGO and Virgo observed binary black hole inspiral events through spacetime ripple distortion.",
            },
        ]
        retriever.index_documents(docs)

        res = retriever.retrieve("How does diffusion tensor imaging assess infant white matter?")
        self.assertGreater(len(res["results"]), 0)
        top_hit = res["results"][0]
        self.assertEqual(top_hit["doc_id"], "med_01")
        self.assertIn("assembled_context", top_hit)
        self.assertLessEqual(res["total_tokens_delivered"], 120)


if __name__ == "__main__":
    unittest.main()
