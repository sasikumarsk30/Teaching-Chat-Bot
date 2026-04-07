"""
Unit tests for app.services.document_processing.chunking_service

Tests all chunking strategies (semantic, paragraph, fixed),
overlap handling, edge cases, and metadata generation.
"""

import pytest
from unittest.mock import patch, MagicMock
from app.services.document_processing.chunking_service import ChunkingService


@pytest.fixture
def chunking_service():
    """Create a ChunkingService with controlled settings."""
    with patch("app.services.document_processing.chunking_service.get_settings") as mock_settings:
        settings = MagicMock()
        settings.chunk_size = 500
        settings.chunk_overlap = 50
        mock_settings.return_value = settings
        service = ChunkingService()
    return service


# ── Semantic Chunking ────────────────────────────────────────

class TestSemanticChunking:
    def test_creates_chunks_from_paragraphs(self, chunking_service, sample_text):
        chunks = chunking_service.chunk_document(
            text=sample_text,
            document_id="doc-001",
            strategy="semantic",
        )
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk["document_id"] == "doc-001"
            assert chunk["content"].strip()

    def test_chunk_metadata_fields(self, chunking_service, sample_text):
        chunks = chunking_service.chunk_document(
            text=sample_text,
            document_id="doc-001",
            strategy="semantic",
        )
        for chunk in chunks:
            assert "id" in chunk
            assert "document_id" in chunk
            assert "sequence" in chunk
            assert "content" in chunk
            assert "chunk_size" in chunk
            assert "start_char" in chunk
            assert "end_char" in chunk
            assert "metadata" in chunk
            assert chunk["metadata"]["strategy"] == "semantic"

    def test_sequential_ordering(self, chunking_service, sample_text):
        chunks = chunking_service.chunk_document(
            text=sample_text,
            document_id="doc-001",
            strategy="semantic",
        )
        sequences = [c["sequence"] for c in chunks]
        assert sequences == list(range(len(chunks)))

    def test_all_content_preserved(self, chunking_service, sample_text):
        """Verify no content is lost during chunking."""
        chunks = chunking_service.chunk_document(
            text=sample_text,
            document_id="doc-001",
            strategy="semantic",
        )
        all_content = " ".join(c["content"] for c in chunks)
        # Each paragraph should appear in at least one chunk
        paragraphs = sample_text.split("\n\n")
        for para in paragraphs:
            # Check that at least most of the paragraph appears somewhere
            assert para[:50] in all_content or para[-50:] in all_content

    def test_respects_chunk_size_limit(self, chunking_service, sample_text):
        chunks = chunking_service.chunk_document(
            text=sample_text,
            document_id="doc-001",
            strategy="semantic",
            chunk_size=200,
        )
        for chunk in chunks:
            assert chunk["chunk_size"] <= 250  # Allow small overflow for semantic boundaries

    def test_large_paragraph_splits_by_sentences(self, chunking_service):
        """A single paragraph larger than chunk_size should be split on sentences."""
        large_para = (
            "This is the first sentence about neural networks. "
            "This is the second sentence about deep learning. "
            "This is the third sentence about convolutional networks. "
            "This is the fourth sentence about recurrent networks. "
            "This is the fifth sentence about attention mechanisms. "
            "This is the sixth sentence about transformers."
        )
        chunks = chunking_service.chunk_document(
            text=large_para,
            document_id="doc-001",
            strategy="semantic",
            chunk_size=150,
        )
        assert len(chunks) > 1


# ── Paragraph Chunking ──────────────────────────────────────

class TestParagraphChunking:
    def test_paragraphs_chunked(self, chunking_service, sample_text):
        chunks = chunking_service.chunk_document(
            text=sample_text,
            document_id="doc-001",
            strategy="paragraph",
        )
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk["metadata"]["strategy"] == "paragraph"

    def test_merges_small_paragraphs(self, chunking_service):
        """Small paragraphs should be merged when they fit within chunk_size."""
        text = "Short para one.\n\nShort para two.\n\nShort para three."
        chunks = chunking_service.chunk_document(
            text=text,
            document_id="doc-001",
            strategy="paragraph",
            chunk_size=500,
        )
        # All three short paragraphs should be merged into one chunk
        assert len(chunks) == 1


# ── Fixed-Size Chunking ─────────────────────────────────────

class TestFixedChunking:
    def test_fixed_chunking_creates_chunks(self, chunking_service, sample_text):
        chunks = chunking_service.chunk_document(
            text=sample_text,
            document_id="doc-001",
            strategy="fixed",
            chunk_size=200,
        )
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk["metadata"]["strategy"] == "fixed"

    def test_fixed_chunks_respect_size(self, chunking_service, sample_text):
        chunk_size = 200
        chunks = chunking_service.chunk_document(
            text=sample_text,
            document_id="doc-001",
            strategy="fixed",
            chunk_size=chunk_size,
        )
        for chunk in chunks:
            assert chunk["chunk_size"] <= chunk_size

    def test_fixed_overlap(self, chunking_service):
        text = "A" * 500
        chunks = chunking_service.chunk_document(
            text=text,
            document_id="doc-001",
            strategy="fixed",
            chunk_size=200,
            chunk_overlap=50,
        )
        assert len(chunks) > 2
        # Check that chunks have overlapping content
        if len(chunks) >= 2:
            c1_end = chunks[0]["content"][-50:]
            c2_start = chunks[1]["content"][:50]
            assert c1_end == c2_start


# ── Edge Cases ───────────────────────────────────────────────

class TestChunkingEdgeCases:
    def test_empty_text_returns_empty(self, chunking_service):
        chunks = chunking_service.chunk_document(
            text="",
            document_id="doc-001",
            strategy="semantic",
        )
        assert chunks == []

    def test_whitespace_only_returns_empty(self, chunking_service):
        chunks = chunking_service.chunk_document(
            text="   \n\n  \t  ",
            document_id="doc-001",
            strategy="semantic",
        )
        assert chunks == []

    def test_unknown_strategy_falls_back_to_semantic(self, chunking_service, sample_text):
        chunks = chunking_service.chunk_document(
            text=sample_text,
            document_id="doc-001",
            strategy="unknown_strategy",
        )
        assert len(chunks) > 0

    def test_single_word(self, chunking_service):
        chunks = chunking_service.chunk_document(
            text="Hello",
            document_id="doc-001",
            strategy="semantic",
        )
        assert len(chunks) == 1
        assert chunks[0]["content"] == "Hello"

    def test_unique_chunk_ids(self, chunking_service, sample_text):
        chunks = chunking_service.chunk_document(
            text=sample_text,
            document_id="doc-001",
            strategy="semantic",
        )
        ids = [c["id"] for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_custom_chunk_size_override(self, chunking_service, sample_text):
        chunks_small = chunking_service.chunk_document(
            text=sample_text,
            document_id="doc-001",
            strategy="fixed",
            chunk_size=100,
        )
        chunks_large = chunking_service.chunk_document(
            text=sample_text,
            document_id="doc-001",
            strategy="fixed",
            chunk_size=1000,
        )
        assert len(chunks_small) > len(chunks_large)
