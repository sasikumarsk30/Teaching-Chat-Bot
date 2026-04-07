"""
Unit tests for app.infrastructure.cache.embedding_cache

Tests LRU eviction, cache hit/miss, batch operations,
content hashing, and statistics.
"""

import pytest
import numpy as np
from app.infrastructure.cache.embedding_cache import EmbeddingCache


@pytest.fixture
def cache():
    """Fresh embedding cache with small max_size for testing eviction."""
    return EmbeddingCache(max_size=5)


@pytest.fixture
def sample_embedding():
    return np.random.randn(384).astype(np.float32)


# ── Single Get/Put ───────────────────────────────────────────

class TestEmbeddingCacheSingle:
    def test_put_and_get(self, cache, sample_embedding):
        cache.put("hello world", sample_embedding)
        result = cache.get("hello world")
        assert result is not None
        np.testing.assert_array_almost_equal(result, sample_embedding)

    def test_cache_miss_returns_none(self, cache):
        assert cache.get("nonexistent text") is None

    def test_same_text_same_key(self, cache, sample_embedding):
        cache.put("test text", sample_embedding)
        result = cache.get("test text")
        assert result is not None

    def test_different_text_different_key(self, cache, sample_embedding):
        cache.put("text A", sample_embedding)
        assert cache.get("text B") is None

    def test_duplicate_put_preserves_original(self, cache):
        emb1 = np.ones(384, dtype=np.float32)
        emb2 = np.zeros(384, dtype=np.float32)
        cache.put("same text", emb1)
        cache.put("same text", emb2)  # move_to_end but no value update
        result = cache.get("same text")
        np.testing.assert_array_equal(result, emb1)


# ── LRU Eviction ────────────────────────────────────────────

class TestEmbeddingCacheLRU:
    def test_evicts_oldest_when_full(self, cache):
        for i in range(6):
            cache.put(f"text_{i}", np.random.randn(384).astype(np.float32))
        # text_0 should have been evicted (cache max_size=5)
        assert cache.get("text_0") is None
        # text_5 should still exist
        assert cache.get("text_5") is not None

    def test_access_refreshes_position(self, cache):
        for i in range(5):
            cache.put(f"text_{i}", np.random.randn(384).astype(np.float32))
        # Access text_0 to move it to end (most recent)
        cache.get("text_0")
        # Add one more item — text_1 should be evicted now
        cache.put("text_new", np.random.randn(384).astype(np.float32))
        assert cache.get("text_0") is not None
        assert cache.get("text_1") is None


# ── Batch Operations ─────────────────────────────────────────

class TestEmbeddingCacheBatch:
    def test_get_batch_empty_cache(self, cache):
        texts = ["a", "b", "c"]
        results, misses = cache.get_batch(texts)
        assert len(results) == 3
        assert all(r is None for r in results)
        assert misses == [0, 1, 2]

    def test_put_and_get_batch(self, cache):
        texts = ["text_a", "text_b", "text_c"]
        embeddings = [np.random.randn(384).astype(np.float32) for _ in texts]
        cache.put_batch(texts, embeddings)

        results, misses = cache.get_batch(texts)
        assert misses == []
        assert all(r is not None for r in results)

    def test_partial_batch_hit(self, cache):
        cache.put("cached_text", np.random.randn(384).astype(np.float32))
        results, misses = cache.get_batch(["cached_text", "new_text"])
        assert results[0] is not None
        assert results[1] is None
        assert misses == [1]


# ── Stats ────────────────────────────────────────────────────

class TestEmbeddingCacheStats:
    def test_initial_stats(self, cache):
        stats = cache.stats
        assert stats["size"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate_pct"] == 0.0

    def test_hit_count_increments(self, cache, sample_embedding):
        cache.put("test", sample_embedding)
        cache.get("test")
        cache.get("test")
        assert cache.stats["hits"] == 2

    def test_miss_count_increments(self, cache):
        cache.get("miss_1")
        cache.get("miss_2")
        assert cache.stats["misses"] == 2

    def test_hit_rate_calculation(self, cache, sample_embedding):
        cache.put("exist", sample_embedding)
        cache.get("exist")  # hit
        cache.get("nope")   # miss
        stats = cache.stats
        assert stats["hit_rate_pct"] == 50.0

    def test_clear_resets_stats(self, cache, sample_embedding):
        cache.put("test", sample_embedding)
        cache.get("test")
        cache.clear()
        stats = cache.stats
        assert stats["size"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0
