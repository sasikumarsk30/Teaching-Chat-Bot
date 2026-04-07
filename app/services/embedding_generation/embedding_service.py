"""
Embedding Service

Generates vector embeddings for text chunks
using open-source sentence-transformers models.
"""

import logging
import uuid
from datetime import datetime
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from app.core.config import get_settings, MODELS_DIR
from app.infrastructure.cache.embedding_cache import get_embedding_cache

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Generates embeddings using sentence-transformers."""

    def __init__(self):
        self.settings = get_settings()
        self.model_name = self.settings.embedding_model_name
        self.dimension = self.settings.embedding_dimension
        self.batch_size = self.settings.embedding_batch_size
        self.cache = get_embedding_cache()
        self._model = None
        self._executor = ThreadPoolExecutor(max_workers=2)
        logger.info(
            f"EmbeddingService initialized | model={self.model_name} "
            f"dim={self.dimension} batch_size={self.batch_size}"
        )

    @property
    def model(self):
        """Lazy-load the embedding model on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            cache_dir = str(MODELS_DIR / "embeddings")
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(
                self.model_name, cache_folder=cache_dir
            )
            logger.info(f"Embedding model loaded: {self.model_name}")
        return self._model

    async def generate_embeddings(
        self, texts: list[str]
    ) -> list[np.ndarray]:
        """
        Generate embeddings for a list of texts.
        Uses cache for previously seen texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of numpy arrays (embeddings) in the same order as input.
        """
        if not texts:
            return []

        # Check cache first
        cached_results, miss_indices = self.cache.get_batch(texts)

        if not miss_indices:
            logger.info(f"All {len(texts)} embeddings served from cache")
            return [r for r in cached_results if r is not None]

        # Generate embeddings for cache misses
        texts_to_embed = [texts[i] for i in miss_indices]
        logger.info(
            f"Generating embeddings | total={len(texts)} "
            f"cached={len(texts) - len(miss_indices)} "
            f"to_generate={len(miss_indices)}"
        )

        # Run in thread pool to avoid blocking async loop
        import asyncio

        loop = asyncio.get_event_loop()
        new_embeddings = await loop.run_in_executor(
            self._executor,
            self._encode_batch,
            texts_to_embed,
        )

        # Update cache
        self.cache.put_batch(texts_to_embed, new_embeddings)

        # Merge cached and new results
        final_results: list[np.ndarray] = []
        new_idx = 0
        for i in range(len(texts)):
            if cached_results[i] is not None:
                final_results.append(cached_results[i])
            else:
                final_results.append(new_embeddings[new_idx])
                new_idx += 1

        return final_results

    def _encode_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Synchronous batch encoding using the model."""
        all_embeddings: list[np.ndarray] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            embeddings = self.model.encode(
                batch,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            all_embeddings.extend(embeddings)

            logger.info(
                f"Encoded batch {i // self.batch_size + 1} "
                f"({len(batch)} texts)"
            )

        return all_embeddings

    async def generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate a single embedding for a search query."""
        results = await self.generate_embeddings([query])
        return results[0]

    def build_vector_records(
        self,
        chunks: list[dict],
        embeddings: list[np.ndarray],
    ) -> list[dict]:
        """
        Pair chunks with their embeddings into vector records
        suitable for storage.
        """
        records = []
        for chunk, embedding in zip(chunks, embeddings):
            records.append({
                "id": str(uuid.uuid4()),
                "chunk_id": chunk["id"],
                "embedding": embedding.tolist(),
                "vector_dim": len(embedding),
                "model_name": self.model_name,
                "created_at": datetime.utcnow(),
            })
        return records

    def get_model_info(self) -> dict:
        """Return details about the loaded model."""
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "batch_size": self.batch_size,
            "cache_stats": self.cache.stats,
        }


# ── Module-level factory ─────────────────────────────────────

_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
