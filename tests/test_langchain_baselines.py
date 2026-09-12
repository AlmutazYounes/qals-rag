"""
Unit tests for LangChain chunking baselines:
- LangChainRecursiveRetriever (500c, 1000c)
- LangChainParentDocRetriever
- LangChainHybridRetriever
"""

import unittest
from sentence_transformers import SentenceTransformer
from toporag.langchain_baselines import (
    LangChainRecursiveRetriever,
    LangChainParentDocRetriever,
    LangChainHybridRetriever,
)


class TestLangChainBaselines(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = SentenceTransformer("all-MiniLM-L6-v2")
        cls.docs = [
            {
                "id": "doc_1",
                "title": "Diffusion MRI in Neonates",
                "text": "Diffusion tensor imaging evaluates cerebral white matter development in newborns. Microstructural changes reflect myelination and axonal organization during gestation. Premature birth often disrupts these developmental pathways.",
            },
            {
                "id": "doc_2",
                "title": "Gravitational Wave Astronomy",
                "text": "LIGO and Virgo detectors observe gravitational waves from coalescing binary black holes. General relativity predicts specific chirping waveforms during inspiral and ringdown.",
            },
        ]

    def test_recursive_splitter_retrieval(self):
        retriever = LangChainRecursiveRetriever(self.model, chunk_size=200, chunk_overlap=20)
        retriever.index_documents(self.docs)
        self.assertGreater(len(retriever.chunks), 0)

        hits = retriever.retrieve("cerebral white matter and brain imaging", k=2)
        self.assertGreater(len(hits), 0)
        self.assertEqual(hits[0]["doc_id"], "doc_1")

    def test_parent_doc_retrieval(self):
        retriever = LangChainParentDocRetriever(self.model, child_chunk_size=100, child_chunk_overlap=10)
        retriever.index_documents(self.docs)
        self.assertGreater(len(retriever.children), 0)

        hits = retriever.retrieve("black hole binary merger waves", k=2)
        self.assertGreater(len(hits), 0)
        self.assertEqual(hits[0]["doc_id"], "doc_2")
        # Confirms parent document text is returned
        self.assertIn("LIGO and Virgo", hits[0]["text"])

    def test_hybrid_retrieval(self):
        dense = LangChainRecursiveRetriever(self.model, chunk_size=200, chunk_overlap=20)
        dense.index_documents(self.docs)
        from toporag.real_baselines import RealBM25Retriever
        bm25 = RealBM25Retriever()
        bm25.index_documents(self.docs)
        hybrid = LangChainHybridRetriever(dense, bm25)

        hits = hybrid.retrieve("gravitational wave chirp waveform", k=2)
        self.assertGreater(len(hits), 0)
        self.assertEqual(hits[0]["doc_id"], "doc_2")


if __name__ == "__main__":
    unittest.main()
