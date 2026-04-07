"""
Unit tests for app.services.embedding_generation.similarity_search

Tests cosine similarity computation, top-k filtering,
threshold filtering, and document-scoped search.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from app.services.embedding_generation.similarity_search import SimilaritySearch


@pytest.fixture
def similarity_search():
    """SimilaritySearch with mocked parquet manager."""
    with patch("app.services.embedding_generation.similarity_search.get_settings") as mock_settings:
        settings = MagicMock()
        settings.search_top_k = 3
        settings.similarity_threshold = 0.1
        mock_settings.return_value = settings

        with patch("app.services.embedding_generation.similarity_search.get_parquet_manager") as mock_parquet:
            parquet = MagicMock()
            mock_parquet.return_value = parquet

            search = SimilaritySearch()
            search.parquet = parquet
            yield search


def _make_vectors_df(n=10, dimension=384, seed=42):
    """Create a DataFrame with fake chunk data and embeddings."""
    np.random.seed(seed)
    data = {
        "id": [f"chunk_{i}" for i in range(n)],
        "document_id": [f"doc_{i % 3}" for i in range(n)],
        "content": [f"Sample content for chunk {i}" for i in range(n)],
        "sequence": list(range(n)),
        "embedding": [np.random.randn(dimension).astype(np.float32).tolist() for _ in range(n)],
        "metadata": [{"strategy": "semantic"} for _ in range(n)],
    }
    return pd.DataFrame(data)


# ── Cosine Similarity ────────────────────────────────────────

class TestCosineSimilarity:
    def test_identical_vectors_score_one(self):
        v = np.array([1.0, 2.0, 3.0])
        score = SimilaritySearch._cosine_similarity(v, v)
        assert abs(score - 1.0) < 1e-6

    def test_orthogonal_vectors_score_zero(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        score = SimilaritySearch._cosine_similarity(a, b)
        assert abs(score) < 1e-6

    def test_opposite_vectors_score_negative(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        score = SimilaritySearch._cosine_similarity(a, b)
        assert score < 0

    def test_zero_vector_returns_zero(self):
        a = np.array([1.0, 2.0, 3.0])
        zero = np.array([0.0, 0.0, 0.0])
        assert SimilaritySearch._cosine_similarity(a, zero) == 0.0


# ── Search Function ──────────────────────────────────────────

class TestSimilaritySearchFunction:
    def test_returns_top_k_results(self, similarity_search):
        df = _make_vectors_df(n=10)
        similarity_search.parquet.read_all_vectors_with_content.return_value = df

        np.random.seed(99)
        query_emb = np.random.randn(384).astype(np.float32)

        results = similarity_search.search(query_emb, top_k=3)
        assert len(results) <= 3

    def test_results_sorted_by_score(self, similarity_search):
        df = _make_vectors_df(n=10)
        similarity_search.parquet.read_all_vectors_with_content.return_value = df

        np.random.seed(99)
        query_emb = np.random.randn(384).astype(np.float32)

        results = similarity_search.search(query_emb, top_k=5)
        if len(results) >= 2:
            scores = [r["similarity_score"] for r in results]
            assert scores == sorted(scores, reverse=True)

    def test_includes_required_fields(self, similarity_search):
        df = _make_vectors_df(n=5)
        similarity_search.parquet.read_all_vectors_with_content.return_value = df

        np.random.seed(99)
        query_emb = np.random.randn(384).astype(np.float32)

        results = similarity_search.search(query_emb, top_k=2)
        if results:
            r = results[0]
            assert "chunk_id" in r
            assert "document_id" in r
            assert "content" in r
            assert "similarity_score" in r
            assert isinstance(r["similarity_score"], float)

    def test_empty_store_returns_empty(self, similarity_search):
        similarity_search.parquet.read_all_vectors_with_content.return_value = pd.DataFrame()

        query_emb = np.random.randn(384).astype(np.float32)
        results = similarity_search.search(query_emb)
        assert results == []

    def test_filters_by_document_id(self, similarity_search):
        df = _make_vectors_df(n=10)
        similarity_search.parquet.read_all_vectors_with_content.return_value = df

        query_emb = np.random.randn(384).astype(np.float32)
        results = similarity_search.search(query_emb, top_k=10, document_id="doc_0")

        for r in results:
            assert r["document_id"] == "doc_0"

    def test_threshold_filtering(self, similarity_search):
        """Results below the threshold should be excluded."""
        similarity_search.threshold = 0.99  # Very high threshold
        df = _make_vectors_df(n=5)
        similarity_search.parquet.read_all_vectors_with_content.return_value = df

        np.random.seed(99)
        query_emb = np.random.randn(384).astype(np.float32)

        results = similarity_search.search(query_emb, top_k=5)
        # With threshold=0.99, random vectors are unlikely to match
        for r in results:
            assert r["similarity_score"] >= 0.99

    def test_no_documents_for_given_id(self, similarity_search):
        df = _make_vectors_df(n=5)
        similarity_search.parquet.read_all_vectors_with_content.return_value = df

        query_emb = np.random.randn(384).astype(np.float32)
        results = similarity_search.search(query_emb, document_id="nonexistent_doc")
        assert results == []
