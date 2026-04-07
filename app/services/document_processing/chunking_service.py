"""
Chunking Service

Splits document text into semantically meaningful chunks
using multiple strategies: semantic, fixed, and paragraph-based.
"""

import logging
import uuid
from datetime import datetime
from typing import Optional

from app.core.config import get_settings
from app.core.constants import (
    CHUNKING_STRATEGY_SEMANTIC,
    CHUNKING_STRATEGY_FIXED,
    CHUNKING_STRATEGY_PARAGRAPH,
)
from app.utils.text_utils import split_sentences, split_paragraphs, extract_heading_structure

logger = logging.getLogger(__name__)


class ChunkingService:
    """Splits text into chunks with configurable strategies."""

    def __init__(self):
        self.settings = get_settings()
        logger.info(
            f"ChunkingService initialized | default_size={self.settings.chunk_size} "
            f"overlap={self.settings.chunk_overlap}"
        )

    def chunk_document(
        self,
        text: str,
        document_id: str,
        strategy: str = CHUNKING_STRATEGY_SEMANTIC,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> list[dict]:
        """
        Split document text into chunks.

        Args:
            text: Full document text.
            document_id: ID of the source document.
            strategy: Chunking strategy to use.
            chunk_size: Override default chunk size (characters).
            chunk_overlap: Override default overlap (characters).

        Returns:
            List of chunk dicts with id, document_id, sequence, content, metadata.
        """
        size = chunk_size or self.settings.chunk_size
        overlap = chunk_overlap or self.settings.chunk_overlap

        if not text or not text.strip():
            logger.warning(f"Empty text for document {document_id}, no chunks created")
            return []

        logger.info(
            f"Chunking document {document_id} | strategy={strategy} "
            f"size={size} overlap={overlap} text_len={len(text)}"
        )

        if strategy == CHUNKING_STRATEGY_SEMANTIC:
            raw_chunks = self._chunk_semantic(text, size, overlap)
        elif strategy == CHUNKING_STRATEGY_PARAGRAPH:
            raw_chunks = self._chunk_by_paragraph(text, size, overlap)
        elif strategy == CHUNKING_STRATEGY_FIXED:
            raw_chunks = self._chunk_fixed(text, size, overlap)
        else:
            logger.warning(f"Unknown strategy '{strategy}', falling back to semantic")
            raw_chunks = self._chunk_semantic(text, size, overlap)

        # Build structured chunk records
        chunks = []
        for i, (content, start_char, end_char) in enumerate(raw_chunks):
            chunk = {
                "id": str(uuid.uuid4()),
                "document_id": document_id,
                "sequence": i,
                "content": content,
                "chunk_size": len(content),
                "start_char": start_char,
                "end_char": end_char,
                "created_at": datetime.utcnow(),
                "metadata": {
                    "strategy": strategy,
                    "original_chunk_size": size,
                    "overlap": overlap,
                },
            }
            chunks.append(chunk)

        logger.info(
            f"Chunking complete | doc={document_id} chunks={len(chunks)}"
        )
        return chunks

    # ── Semantic Chunking ────────────────────────────────────

    def _chunk_semantic(
        self, text: str, chunk_size: int, overlap: int
    ) -> list[tuple[str, int, int]]:
        """
        Chunk by respecting semantic boundaries (paragraphs → sentences).

        Priority:
        1. Split into paragraphs
        2. If a paragraph exceeds chunk_size, split by sentences
        3. Merge small consecutive paragraphs until chunk_size
        4. Apply overlap between chunks
        """
        paragraphs = split_paragraphs(text)
        chunks: list[tuple[str, int, int]] = []

        current_chunk = ""
        current_start = 0
        position = 0

        for para in paragraphs:
            para_start = text.find(para, position)
            if para_start == -1:
                para_start = position

            if len(para) > chunk_size:
                # Paragraph too large → split further by sentences
                if current_chunk:
                    chunks.append((
                        current_chunk.strip(),
                        current_start,
                        current_start + len(current_chunk.strip()),
                    ))
                    current_chunk = ""

                sentence_chunks = self._split_large_text_by_sentences(
                    para, chunk_size, overlap, para_start
                )
                chunks.extend(sentence_chunks)
                position = para_start + len(para)
                current_start = position
                continue

            # Try to merge into current chunk
            candidate = (current_chunk + "\n\n" + para).strip() if current_chunk else para
            if len(candidate) <= chunk_size:
                if not current_chunk:
                    current_start = para_start
                current_chunk = candidate
            else:
                # Flush current chunk
                if current_chunk:
                    chunks.append((
                        current_chunk.strip(),
                        current_start,
                        current_start + len(current_chunk.strip()),
                    ))

                    # Apply overlap from end of last chunk
                    if overlap > 0:
                        overlap_text = current_chunk.strip()[-overlap:]
                        current_chunk = overlap_text + "\n\n" + para
                        current_start = para_start - len(overlap_text)
                    else:
                        current_chunk = para
                        current_start = para_start
                else:
                    current_chunk = para
                    current_start = para_start

            position = para_start + len(para)

        # Flush remaining
        if current_chunk.strip():
            chunks.append((
                current_chunk.strip(),
                current_start,
                current_start + len(current_chunk.strip()),
            ))

        return chunks

    def _split_large_text_by_sentences(
        self, text: str, chunk_size: int, overlap: int, base_offset: int
    ) -> list[tuple[str, int, int]]:
        """Split a large paragraph by sentence boundaries."""
        sentences = split_sentences(text)
        chunks: list[tuple[str, int, int]] = []

        current_chunk = ""
        current_start = base_offset

        for sentence in sentences:
            candidate = (current_chunk + " " + sentence).strip() if current_chunk else sentence

            if len(candidate) <= chunk_size:
                if not current_chunk:
                    sent_pos = text.find(sentence)
                    current_start = base_offset + (sent_pos if sent_pos != -1 else 0)
                current_chunk = candidate
            else:
                if current_chunk:
                    chunks.append((
                        current_chunk.strip(),
                        current_start,
                        current_start + len(current_chunk.strip()),
                    ))
                    # Overlap
                    if overlap > 0:
                        overlap_text = current_chunk.strip()[-overlap:]
                        current_chunk = overlap_text + " " + sentence
                    else:
                        current_chunk = sentence
                else:
                    # Single sentence larger than chunk_size → force-split
                    chunks.extend(
                        self._force_split(sentence, chunk_size, overlap, current_start)
                    )
                    current_chunk = ""

                sent_pos = text.find(sentence)
                current_start = base_offset + (sent_pos if sent_pos != -1 else 0)

        if current_chunk.strip():
            chunks.append((
                current_chunk.strip(),
                current_start,
                current_start + len(current_chunk.strip()),
            ))

        return chunks

    # ── Paragraph Chunking ───────────────────────────────────

    def _chunk_by_paragraph(
        self, text: str, chunk_size: int, overlap: int
    ) -> list[tuple[str, int, int]]:
        """Chunk by paragraph boundaries, merging small paragraphs."""
        paragraphs = split_paragraphs(text)
        chunks: list[tuple[str, int, int]] = []

        current_chunk = ""
        current_start = 0

        for para in paragraphs:
            para_start = text.find(para)
            if para_start == -1:
                para_start = 0

            candidate = (current_chunk + "\n\n" + para).strip() if current_chunk else para

            if len(candidate) <= chunk_size:
                if not current_chunk:
                    current_start = para_start
                current_chunk = candidate
            else:
                if current_chunk:
                    chunks.append((
                        current_chunk.strip(),
                        current_start,
                        current_start + len(current_chunk.strip()),
                    ))
                current_chunk = para
                current_start = para_start

        if current_chunk.strip():
            chunks.append((
                current_chunk.strip(),
                current_start,
                current_start + len(current_chunk.strip()),
            ))

        return chunks

    # ── Fixed-Size Chunking ──────────────────────────────────

    def _chunk_fixed(
        self, text: str, chunk_size: int, overlap: int
    ) -> list[tuple[str, int, int]]:
        """Simple fixed-size chunking with overlap."""
        return self._force_split(text, chunk_size, overlap, 0)

    def _force_split(
        self, text: str, chunk_size: int, overlap: int, offset: int
    ) -> list[tuple[str, int, int]]:
        """Force-split text into fixed-size chunks."""
        chunks: list[tuple[str, int, int]] = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]
            chunks.append((chunk_text, offset + start, offset + end))

            step = chunk_size - overlap
            if step <= 0:
                step = chunk_size  # Prevent infinite loop
            start += step

        return chunks


# ── Module-level factory ─────────────────────────────────────

_chunking_service: Optional[ChunkingService] = None


def get_chunking_service() -> ChunkingService:
    global _chunking_service
    if _chunking_service is None:
        _chunking_service = ChunkingService()
    return _chunking_service
