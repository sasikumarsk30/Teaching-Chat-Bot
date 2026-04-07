"""
Similarity Search

Performs semantic search over stored embeddings
using cosine similarity computed in DuckDB or NumPy.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from app.core.config import get_settings
from app.infrastructure.data_access.parquet_manager import get_parquet_manager

logger = logging.getLogger(__name__)


class SimilaritySearch:
    """Semantic search over chunk embeddings."""

    def __init__(self):
        self.settings = get_settings()
        self.parquet = get_parquet_manager()
        self.default_top_k = self.settings.search_top_k
        self.threshold = self.settings.similarity_threshold
        logger.info(
            f"SimilaritySearch initialized | top_k={self.default_top_k} "
            f"threshold={self.threshold}"
        )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: Optional[int] = None,
        document_id: Optional[str] = None,
    ) -> list[dict]:
        """
        Find the most similar chunks to a query embedding.

        Args:
            query_embedding: The query vector.
            top_k: Number of results to return.
            document_id: Scope search to a specific document.

        Returns:
            List of dicts with chunk metadata + similarity_score,
            sorted by descending similarity.
        """
        k = top_k or self.default_top_k

        # Load all vectors with content
        df = self.parquet.read_all_vectors_with_content()

        if df.empty:
            logger.warning("No vectors in store — returning empty results")
            return []

        # Filter by document if specified
        if document_id and "document_id" in df.columns:
            df = df[df["document_id"] == document_id]
            if df.empty:
                logger.warning(f"No vectors for document {document_id}")
                return []

        # Compute cosine similarity
        similarities = df["embedding"].apply(
            lambda emb: self._cosine_similarity(
                query_embedding, np.array(emb, dtype=np.float32)
            )
        )
        df = df.copy()
        df["similarity_score"] = similarities

        # Filter by threshold
        df = df[df["similarity_score"] >= self.threshold]

        # Sort and take top-k
        df = df.sort_values("similarity_score", ascending=False).head(k)

        results = []
        for _, row in df.iterrows():
            result = {
                "chunk_id": row.get("id", ""),
                "document_id": row.get("document_id", ""),
                "content": row.get("content", ""),
                "sequence": row.get("sequence", 0),
                "similarity_score": float(row["similarity_score"]),
                "metadata": row.get("metadata", {}),
            }
            results.append(result)

        logger.info(
            f"Similarity search | candidates={len(df)} "
            f"returned={len(results)} top_score="
            f"{results[0]['similarity_score']:.4f}" if results else
            f"Similarity search | no results above threshold"
        )

        return results

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))


# ── Module-level factory ─────────────────────────────────────

_similarity_search: Optional[SimilaritySearch] = None


def get_similarity_search() -> SimilaritySearch:
    global _similarity_search
    if _similarity_search is None:
        _similarity_search = SimilaritySearch()
    return _similarity_search
