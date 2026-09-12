"""
TopoRAG: Manifold-Calibrated Adaptive Multi-Granularity Retrieval Engine
"""

from toporag.embeddings import HashFeatureEmbedder, DenseEmbedder
from toporag.chunking import Chunk, HierarchicalChunker
from toporag.calibration import ManifoldCalibrator
from toporag.dynamic_cutoff import DynamicCutoffSelector
from toporag.retriever import TopoRAGRetriever
from toporag.baselines import FlatDenseRetriever, BM25Retriever

__version__ = "0.1.0"
__all__ = [
    "HashFeatureEmbedder",
    "DenseEmbedder",
    "Chunk",
    "HierarchicalChunker",
    "ManifoldCalibrator",
    "DynamicCutoffSelector",
    "TopoRAGRetriever",
    "FlatDenseRetriever",
    "BM25Retriever",
]
