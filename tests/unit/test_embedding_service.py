"""
Unit tests for app.services.embedding_generation.embedding_service

Tests embedding generation with mocked sentence-transformers model,
cache integration, batch processing, and vector record building.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock, AsyncMock
from app.services.embedding_generation.embedding_service import EmbeddingService
from app.infrastructure.cache.embedding_cache import EmbeddingCache


@pytest.fixture
def embedding_service(mock_embedding_model):
    """EmbeddingService with mocked model and fresh cache."""
    with patch("app.services.embedding_generation.embedding_service.get_settings") as mock_settings:
        settings = MagicMock()
        settings.embedding_model_name = "test-model"
        settings.embedding_dimension = 384
        settings.embedding_batch_size = 16
        mock_settings.return_value = settings

        with patch("app.services.embedding_generation.embedding_service.get_embedding_cache") as mock_cache_fn:
            cache = EmbeddingCache(max_size=100)
            mock_cache_fn.return_value = cache

            service = EmbeddingService()
            service._model = mock_embedding_model
            yield service


# ── Embedding Generation ─────────────────────────────────────

class TestEmbeddingGeneration:
    @pytest.mark.asyncio
    async def test_generates_embeddings(self, embedding_service):
        texts = ["Hello world", "Machine learning is great"]
        embeddings = await embedding_service.generate_embeddings(texts)
        assert len(embeddings) == 2
        for emb in embeddings:
            assert isinstance(emb, np.ndarray)
            assert emb.shape == (384,)

    @pytest.mark.asyncio
    async def test_empty_input_returns_empty(self, embedding_service):
        embeddings = await embedding_service.generate_embeddings([])
        assert embeddings == []

    @pytest.mark.asyncio
    async def test_single_text(self, embedding_service):
        embeddings = await embedding_service.generate_embeddings(["Just one text"])
        assert len(embeddings) == 1

    @pytest.mark.asyncio
    async def test_query_embedding(self, embedding_service):
        embedding = await embedding_service.generate_query_embedding("test query")
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)


# ── Cache Integration ────────────────────────────────────────

class TestEmbeddingCacheIntegration:
    @pytest.mark.asyncio
    async def test_uses_cache_on_repeated_texts(self, embedding_service):
        texts = ["cached text"]
        # First call — generates embedding
        emb1 = await embedding_service.generate_embeddings(texts)
        # Second call — should use cache
        emb2 = await embedding_service.generate_embeddings(texts)
        np.testing.assert_array_equal(emb1[0], emb2[0])

    @pytest.mark.asyncio
    async def test_partial_cache_hit(self, embedding_service):
        # Seed cache with one text
        await embedding_service.generate_embeddings(["already cached"])
        # Now request two texts — one cached, one new
        result = await embedding_service.generate_embeddings(
            ["already cached", "brand new text"]
        )
        assert len(result) == 2


# ── Vector Record Building ───────────────────────────────────

class TestBuildVectorRecords:
    def test_builds_records(self, embedding_service, sample_chunks, sample_embeddings):
        records = embedding_service.build_vector_records(
            sample_chunks[:3], sample_embeddings[:3]
        )
        assert len(records) == 3
        for rec in records:
            assert "id" in rec
            assert "chunk_id" in rec
            assert "embedding" in rec
            assert "vector_dim" in rec
            assert "model_name" in rec
            assert "created_at" in rec
            assert isinstance(rec["embedding"], list)
            assert rec["vector_dim"] == 384

    def test_chunk_ids_matched(self, embedding_service, sample_chunks, sample_embeddings):
        records = embedding_service.build_vector_records(
            sample_chunks[:2], sample_embeddings[:2]
        )
        assert records[0]["chunk_id"] == sample_chunks[0]["id"]
        assert records[1]["chunk_id"] == sample_chunks[1]["id"]


# ── Model Info ───────────────────────────────────────────────

class TestModelInfo:
    def test_get_model_info(self, embedding_service):
        info = embedding_service.get_model_info()
        assert "model_name" in info
        assert "dimension" in info
        assert "batch_size" in info
        assert "cache_stats" in info
