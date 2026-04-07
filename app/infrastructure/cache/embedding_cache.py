"""
Embedding Cache

In-memory LRU cache for recently computed embeddings
to avoid redundant model inference.
"""

import logging
import hashlib
from collections import OrderedDict
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """LRU cache for embedding vectors keyed by content hash."""

    def __init__(self, max_size: int = 5000):
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._max_size = max_size
        self._hits = 0
        self._misses = 0
        logger.info(f"EmbeddingCache initialized | max_size={max_size}")

    @staticmethod
    def _hash_text(text: str) -> str:
        """Compute a sha256 hash of the text for cache key."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> Optional[np.ndarray]:
        """Retrieve a cached embedding or None."""
        key = self._hash_text(text)
        if key in self._cache:
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def put(self, text: str, embedding: np.ndarray) -> None:
        """Store an embedding in the cache."""
        key = self._hash_text(text)
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)  # Remove oldest
            self._cache[key] = embedding

    def get_batch(self, texts: list[str]) -> tuple[list[np.ndarray | None], list[int]]:
        """
        Look up a batch of texts. Returns:
          - list of embeddings (None for misses)
          - list of indices that missed (need embedding)
        """
        results: list[np.ndarray | None] = []
        miss_indices: list[int] = []
        for i, text in enumerate(texts):
            emb = self.get(text)
            results.append(emb)
            if emb is None:
                miss_indices.append(i)
        return results, miss_indices

    def put_batch(self, texts: list[str], embeddings: list[np.ndarray]) -> None:
        """Store a batch of embeddings."""
        for text, emb in zip(texts, embeddings):
            self.put(text, emb)

    def clear(self) -> None:
        """Clear the entire cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        logger.info("EmbeddingCache cleared")

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0.0
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_pct": round(hit_rate, 2),
        }


# ── Module-level singleton ───────────────────────────────────

_embedding_cache: Optional[EmbeddingCache] = None


def get_embedding_cache() -> EmbeddingCache:
    global _embedding_cache
    if _embedding_cache is None:
        _embedding_cache = EmbeddingCache()
    return _embedding_cache
