"""
TopoRAG: Manifold-Calibrated Adaptive Multi-Granularity Retrieval Engine
"""

from toporag.embeddings import HashFeatureEmbedder, DenseEmbedder
from toporag.chunking import Chunk, HierarchicalChunker
from toporag.calibration import ManifoldCalibrator
from toporag.dynamic_cutoff import DynamicCutoffSelector
from toporag.retriever import TopoRAGRetriever
from toporag.baselines import FlatDenseRetriever, BM25Retriever
from toporag.qals_encoder import SentenceMultiVectorEncoder
from toporag.qals_dp import SpanSegmenterDP
from toporag.conformal import ConformalBudgetCalibrator
from toporag.qals_retriever import QALSRetriever
from toporag.real_baselines import RealFlatDenseRetriever, RealParentDocumentRetriever, RealBM25Retriever
from toporag.langchain_baselines import (
    LangChainRecursiveRetriever,
    LangChainParentDocRetriever,
    LangChainHybridRetriever,
)

__version__ = "0.3.0"
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
    "SentenceMultiVectorEncoder",
    "SpanSegmenterDP",
    "ConformalBudgetCalibrator",
    "QALSRetriever",
    "RealFlatDenseRetriever",
    "RealParentDocumentRetriever",
    "RealBM25Retriever",
    "LangChainRecursiveRetriever",
    "LangChainParentDocRetriever",
    "LangChainHybridRetriever",
]
