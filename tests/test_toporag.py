"""
Unit tests for TopoRAG core modules:
- Hierarchical chunking
- HashFeatureEmbedder
- ManifoldCalibrator
- DynamicCutoffSelector
- TopoRAGRetriever end-to-end
"""

import unittest
import numpy as np
from toporag.chunking import HierarchicalChunker
from toporag.embeddings import HashFeatureEmbedder
from toporag.calibration import ManifoldCalibrator
from toporag.dynamic_cutoff import DynamicCutoffSelector
from toporag.retriever import TopoRAGRetriever


class TestTopoRAG(unittest.TestCase):

    def setUp(self):
        self.sample_text = (
            "Vector databases perform similarity search using cosine distance over high-dimensional representations. "
            "However, anisotropy causes embedding vectors to cluster in a narrow cone, leading to hubness. "
            "Hubness makes a small fraction of vectors appear as nearest neighbors for an excessive number of queries. "
            "\n\n"
            "To solve this, TopoRAG introduces manifold calibration and dynamic knee cutoff. "
            "Instead of fixing k=5 or k=10, the dynamic cutoff algorithm detects the natural relevance elbow. "
            "Furthermore, hierarchical chunking bridges granular sentence search with broader contextual passage synthesis. "
            "\n\n"
            "Empirical results demonstrate improved NDCG and a significant reduction in context noise and hallucination."
        )

    def test_hierarchical_chunking(self):
        chunker = HierarchicalChunker(micro_size=80, meso_size=250, macro_size=800)
        chunks = chunker.chunk_document("doc_test", self.sample_text)
        self.assertGreater(len(chunks), 0)
        levels = set(c.level for c in chunks)
        self.assertIn("micro", levels)
        self.assertIn("meso", levels)
        self.assertIn("macro", levels)

        # Check parent-child linkage
        micro_chunks = [c for c in chunks if c.level == "micro"]
        for mc in micro_chunks:
            self.assertIsNotNone(mc.parent_id)

    def test_embeddings(self):
        embedder = HashFeatureEmbedder(dim=256, anisotropy_strength=0.0)
        v1 = embedder.embed("Vector databases and cosine similarity")
        v2 = embedder.embed("High dimensional vector representations")
        v3 = embedder.embed("Completely unrelated baking recipe for chocolate cake")

        self.assertEqual(v1.shape, (256,))
        # v1 and v2 should be closer than v1 and v3
        sim_12 = float(np.dot(v1, v2))
        sim_13 = float(np.dot(v1, v3))
        self.assertGreater(sim_12, sim_13)

    def test_manifold_calibrator(self):
        calibrator = ManifoldCalibrator(k_density=3, lambda_hub=0.4, gamma_hub=0.2)
        # Create synthetic embeddings where point 0 is a hub (close to everything)
        emb = np.random.randn(20, 32).astype(np.float32)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        # Force point 0 to be near center
        emb[0] = np.mean(emb, axis=0)
        emb[0] /= np.linalg.norm(emb[0])

        calibrator.fit(emb)
        self.assertIsNotNone(calibrator.hub_scores)
        self.assertEqual(len(calibrator.hub_scores), 20)

        raw_sims = np.ones(20, dtype=np.float32) * 0.8
        cal_sims = calibrator.calibrate(raw_sims)
        self.assertEqual(len(cal_sims), 20)
        # Hub point should receive higher penalty
        if calibrator.hub_scores[0] > calibrator.hub_scores[1]:
            self.assertLess(cal_sims[0], cal_sims[1])

    def test_dynamic_cutoff(self):
        selector = DynamicCutoffSelector(min_k=1, max_k=8, drop_threshold=0.2)
        # Distinct knee after 2 elements: [0.95, 0.91, 0.45, 0.42, 0.38]
        scores = np.array([0.95, 0.91, 0.45, 0.42, 0.38, 0.20], dtype=np.float32)
        k, diag = selector.select_cutoff(scores)
        self.assertIn(k, [2, 3])  # Should cleanly stop around the knee

    def test_end_to_end_retriever(self):
        embedder = HashFeatureEmbedder(dim=256, anisotropy_strength=0.0)
        retriever = TopoRAGRetriever(embedder=embedder)
        docs = [
            {"id": "doc1", "text": self.sample_text},
            {"id": "doc2", "text": "Cooking lasagna requires fresh pasta sheets, tomato marinara, ricotta, and mozzarella."},
        ]
        retriever.index_documents(docs)

        res = retriever.retrieve("How does hubness affect vector similarity search?")
        self.assertGreater(len(res["results"]), 0)
        top_hit = res["results"][0]
        self.assertEqual(top_hit["doc_id"], "doc1")
        self.assertIn("retrieved_context", top_hit)


if __name__ == "__main__":
    unittest.main()
