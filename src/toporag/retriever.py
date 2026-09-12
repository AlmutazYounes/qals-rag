"""
Core TopoRAG Engine.
Integrates Multi-Scale Chunking, Manifold Calibration, Hierarchical Aggregation,
and Dynamic Cutoff Selection for similarity search.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from toporag.chunking import Chunk, HierarchicalChunker
from toporag.embeddings import DenseEmbedder, HashFeatureEmbedder
from toporag.calibration import ManifoldCalibrator
from toporag.dynamic_cutoff import DynamicCutoffSelector


class TopoRAGRetriever:
    """
    TopoRAG: Manifold-Calibrated Adaptive Multi-Granularity Retrieval Engine.
    
    1. Indexes documents into micro, meso, and macro multi-scale hierarchy.
    2. Embeds micro chunks for high-specificity matching.
    3. Fits Riemannian density and hubness calibration across the embedding gallery.
    4. At query time:
       - Projects query into embedding space.
       - Computes calibrated non-hub similarity scores.
       - Runs Dynamic Knee Cutoff (adaptive top-k) to prevent over/under-retrieval.
       - Expands retrieved micro chunks upward to their parent meso/macro contexts,
         avoiding context fragmentation without losing precision.
    """

    def __init__(
        self,
        embedder=None,
        chunker: Optional[HierarchicalChunker] = None,
        calibrator: Optional[ManifoldCalibrator] = None,
        cutoff_selector: Optional[DynamicCutoffSelector] = None,
        parent_aggregation_threshold: int = 1,
    ):
        self.embedder = embedder or HashFeatureEmbedder(dim=128)
        self.chunker = chunker or HierarchicalChunker()
        self.calibrator = calibrator or ManifoldCalibrator()
        self.cutoff_selector = cutoff_selector or DynamicCutoffSelector()
        self.parent_aggregation_threshold = parent_aggregation_threshold

        # Storage
        self.chunks: List[Chunk] = []
        self.chunk_by_id: Dict[str, Chunk] = {}
        self.micro_chunks: List[Chunk] = []
        self.micro_embeddings: Optional[np.ndarray] = None
        self.is_indexed: bool = False

    def index_documents(self, documents: List[Dict[str, str]]):
        """
        documents: list of dicts with 'id' and 'text', plus optional 'metadata'
        """
        all_chunks = []
        for doc in documents:
            doc_id = doc.get("id", f"doc_{len(all_chunks)}")
            text = doc.get("text", "")
            meta = doc.get("metadata", {})
            chunks = self.chunker.chunk_document(doc_id, text, metadata=meta)
            all_chunks.extend(chunks)

        self.chunks = all_chunks
        self.chunk_by_id = {c.id: c for c in all_chunks}
        self.micro_chunks = [c for c in all_chunks if c.level == "micro"]

        if not self.micro_chunks:
            # Fallback if text was very short
            self.micro_chunks = self.chunks

        texts = [c.text for c in self.micro_chunks]
        self.micro_embeddings = self.embedder.embed(texts)

        # Calibrate manifold (estimate hubness and local density)
        self.calibrator.fit(self.micro_embeddings)
        self.is_indexed = True

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        expand_to_parent: bool = True,
        return_diagnostics: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes query retrieval:
          - Calibrated similarity search
          - Dynamic cutoff
          - Multi-scale context expansion
        """
        if not self.is_indexed or self.micro_embeddings is None:
            raise ValueError("TopoRAG index is empty. Call index_documents() first.")

        query_emb = self.embedder.embed(query)
        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)

        # 1. Raw cosine similarities
        raw_sims = (query_emb @ self.micro_embeddings.T)[0]  # (N,)

        # 2. Manifold calibration (mitigates hubness & anisotropy)
        calibrated_sims = self.calibrator.calibrate(raw_sims, query_emb=query_emb[0])

        # 3. Sort candidates
        sorted_indices = np.argsort(-calibrated_sims)
        sorted_scores = calibrated_sims[sorted_indices]

        # 4. Adaptive Cutoff
        if k is None:
            selected_k, diag = self.cutoff_selector.select_cutoff(sorted_scores)
        else:
            selected_k = min(k, len(sorted_scores))
            diag = {"selected_k": selected_k, "reason": "user_fixed_k"}

        top_indices = sorted_indices[:selected_k]
        retrieved_micro = [self.micro_chunks[idx] for idx in top_indices]
        retrieved_scores = [float(sorted_scores[i]) for i in range(selected_k)]
        raw_scores_retrieved = [float(raw_sims[idx]) for idx in top_indices]

        # 5. Multi-Scale Context Expansion
        # If expand_to_parent is True, resolve micro chunks to their parent (meso) context,
        # deduplicating parents when multiple micro chunks hit the same passage.
        final_passages = []
        parent_seen = set()

        for micro, cal_score, raw_score in zip(retrieved_micro, retrieved_scores, raw_scores_retrieved):
            item = {
                "micro_id": micro.id,
                "micro_text": micro.text,
                "calibrated_score": round(cal_score, 4),
                "raw_score": round(raw_score, 4),
                "doc_id": micro.doc_id,
                "metadata": micro.metadata,
            }

            if expand_to_parent and micro.parent_id and micro.parent_id in self.chunk_by_id:
                parent_chunk = self.chunk_by_id[micro.parent_id]
                item["retrieved_context"] = parent_chunk.text
                item["context_level"] = parent_chunk.level
                item["context_id"] = parent_chunk.id

                # Deduplicate output contexts while preserving highest score
                if parent_chunk.id not in parent_seen:
                    parent_seen.add(parent_chunk.id)
                    final_passages.append(item)
            else:
                item["retrieved_context"] = micro.text
                item["context_level"] = micro.level
                item["context_id"] = micro.id
                final_passages.append(item)

        result = {
            "query": query,
            "results": final_passages,
            "total_micro_hits": selected_k,
            "unique_contexts_returned": len(final_passages),
        }
        if return_diagnostics:
            result["diagnostics"] = diag
            result["diagnostics"]["total_gallery_size"] = len(self.micro_chunks)

        return result
