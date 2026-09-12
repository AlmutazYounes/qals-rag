"""
Hierarchical text chunking with parent-child relationship tracking.
Provides micro-chunks (sentences/sub-paragraphs) linked to meso/macro parent passages.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import re


@dataclass
class Chunk:
    id: str
    text: str
    level: str  # 'micro', 'meso', 'macro'
    parent_id: Optional[str] = None
    child_ids: List[str] = field(default_factory=list)
    doc_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    char_start: int = 0
    char_end: int = 0


class HierarchicalChunker:
    """
    Constructs a multi-scale hierarchy:
      - Macro: Full section / document unit (~500-1000 tokens / 2000-4000 chars)
      - Meso: Paragraph / structural thought (~150-300 tokens / 600-1200 chars)
      - Micro: Sentences / granular proposition (~30-80 tokens / 120-300 chars)
    """

    def __init__(
        self,
        micro_size: int = 200,
        meso_size: int = 800,
        macro_size: int = 2400,
        overlap: int = 40,
    ):
        self.micro_size = micro_size
        self.meso_size = meso_size
        self.macro_size = macro_size
        self.overlap = overlap

    def _split_into_sentences(self, text: str) -> List[str]:
        # Split on sentence boundaries while keeping sentences intact
        splits = re.split(r"(?<=[.?!])\s+", text.strip())
        return [s.strip() for s in splits if s.strip()]

    def chunk_document(self, doc_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        meta = metadata or {}
        chunks: List[Chunk] = []

        # 1. Macro chunks: divide text into macro windows
        macro_step = max(self.macro_size - self.overlap, self.macro_size // 2)
        raw_len = len(text)
        
        macro_indices = []
        start = 0
        while start < raw_len:
            end = min(start + self.macro_size, raw_len)
            macro_indices.append((start, end))
            if end >= raw_len:
                break
            start += macro_step

        for m_idx, (m_start, m_end) in enumerate(macro_indices):
            macro_text = text[m_start:m_end]
            macro_id = f"{doc_id}_macro_{m_idx}"
            macro_chunk = Chunk(
                id=macro_id,
                text=macro_text,
                level="macro",
                doc_id=doc_id,
                metadata=meta,
                char_start=m_start,
                char_end=m_end,
            )
            chunks.append(macro_chunk)

            # 2. Meso chunks inside this macro
            paragraphs = [p.strip() for p in macro_text.split("\n\n") if p.strip()]
            if not paragraphs:
                paragraphs = [macro_text]

            current_meso_text = ""
            current_meso_start = m_start
            meso_idx = 0

            for p in paragraphs:
                if len(current_meso_text) + len(p) > self.meso_size and current_meso_text:
                    meso_id = f"{macro_id}_meso_{meso_idx}"
                    meso_chunk = Chunk(
                        id=meso_id,
                        text=current_meso_text,
                        level="meso",
                        parent_id=macro_id,
                        doc_id=doc_id,
                        metadata=meta,
                        char_start=current_meso_start,
                        char_end=current_meso_start + len(current_meso_text),
                    )
                    macro_chunk.child_ids.append(meso_id)
                    chunks.append(meso_chunk)

                    # Create micro chunks for this meso chunk
                    self._create_micro_chunks(meso_chunk, chunks)

                    meso_idx += 1
                    current_meso_text = p
                    current_meso_start = m_start + macro_text.find(p)
                else:
                    if current_meso_text:
                        current_meso_text += "\n\n" + p
                    else:
                        current_meso_text = p
                        current_meso_start = m_start + macro_text.find(p)

            if current_meso_text:
                meso_id = f"{macro_id}_meso_{meso_idx}"
                meso_chunk = Chunk(
                    id=meso_id,
                    text=current_meso_text,
                    level="meso",
                    parent_id=macro_id,
                    doc_id=doc_id,
                    metadata=meta,
                    char_start=current_meso_start,
                    char_end=current_meso_start + len(current_meso_text),
                )
                macro_chunk.child_ids.append(meso_id)
                chunks.append(meso_chunk)
                self._create_micro_chunks(meso_chunk, chunks)

        return chunks

    def _create_micro_chunks(self, meso_chunk: Chunk, all_chunks: List[Chunk]):
        sentences = self._split_into_sentences(meso_chunk.text)
        current_micro = ""
        current_start = meso_chunk.char_start
        micro_idx = 0

        for s in sentences:
            if len(current_micro) + len(s) > self.micro_size and current_micro:
                micro_id = f"{meso_chunk.id}_micro_{micro_idx}"
                mc = Chunk(
                    id=micro_id,
                    text=current_micro,
                    level="micro",
                    parent_id=meso_chunk.id,
                    doc_id=meso_chunk.doc_id,
                    metadata=meso_chunk.metadata,
                    char_start=current_start,
                    char_end=current_start + len(current_micro),
                )
                meso_chunk.child_ids.append(micro_id)
                all_chunks.append(mc)
                micro_idx += 1
                current_micro = s
                pos = meso_chunk.text.find(s)
                current_start = meso_chunk.char_start + (pos if pos != -1 else 0)
            else:
                if current_micro:
                    current_micro += " " + s
                else:
                    current_micro = s
                    pos = meso_chunk.text.find(s)
                    current_start = meso_chunk.char_start + (pos if pos != -1 else 0)

        if current_micro:
            micro_id = f"{meso_chunk.id}_micro_{micro_idx}"
            mc = Chunk(
                id=micro_id,
                text=current_micro,
                level="micro",
                parent_id=meso_chunk.id,
                doc_id=meso_chunk.doc_id,
                metadata=meso_chunk.metadata,
                char_start=current_start,
                char_end=current_start + len(current_micro),
            )
            meso_chunk.child_ids.append(micro_id)
            all_chunks.append(mc)
