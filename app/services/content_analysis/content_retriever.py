"""
Content Retriever

Retrieves relevant document chunks based on user queries
by combining semantic search with optional keyword filtering.
"""

import logging
from typing import Optional

import numpy as np

from app.core.config import get_settings
from app.services.embedding_generation.embedding_service import get_embedding_service
from app.services.embedding_generation.similarity_search import get_similarity_search
from app.infrastructure.data_access.duckdb_manager import get_duckdb_manager
from app.core.constants import TABLE_CHUNKS, TABLE_DOCUMENTS

logger = logging.getLogger(__name__)


class ContentRetriever:
    """Retrieves relevant chunks for a user query."""

    def __init__(self):
        self.settings = get_settings()
        self.embedding_service = get_embedding_service()
        self.similarity_search = get_similarity_search()
        self.db = get_duckdb_manager()
        logger.info("ContentRetriever initialized")

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        document_id: Optional[str] = None,
    ) -> list[dict]:
        """
        Retrieve the most relevant chunks for a query.

        1. Generate query embedding.
        2. Perform similarity search.
        3. Enrich results with document metadata.

        Args:
            query: The user's question or topic.
            top_k: Number of results to return.
            document_id: Scope search to one document.

        Returns:
            List of chunk dicts with content, similarity score, and source info.
        """
        k = top_k or self.settings.search_top_k

        # Generate query embedding
        query_embedding = await self.embedding_service.generate_query_embedding(query)

        # Similarity search
        results = self.similarity_search.search(
            query_embedding=query_embedding,
            top_k=k,
            document_id=document_id,
        )

        if not results:
            logger.info(f"No results for query: '{query[:80]}...'")
            return []

        # Enrich with document titles
        enriched = self._enrich_with_document_info(results)

        logger.info(
            f"Retrieved {len(enriched)} chunks for query: '{query[:80]}...' "
            f"top_score={enriched[0]['similarity_score']:.4f}"
        )
        return enriched

    async def retrieve_for_subtopics(
        self,
        subtopics: list[str],
        max_per_subtopic: int = 3,
        document_ids: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Retrieve chunks for multiple subtopics (predefined content mode).

        Args:
            subtopics: List of subtopic strings.
            max_per_subtopic: Max chunks per subtopic.
            document_ids: Scope to specific documents.

        Returns:
            De-duplicated list of chunks covering all subtopics.
        """
        all_results: list[dict] = []
        seen_chunk_ids: set = set()

        for subtopic in subtopics:
            results = await self.retrieve(
                query=subtopic,
                top_k=max_per_subtopic,
                document_id=document_ids[0] if document_ids and len(document_ids) == 1 else None,
            )

            for chunk in results:
                chunk_id = chunk.get("chunk_id", "")
                if chunk_id not in seen_chunk_ids:
                    chunk["subtopic"] = subtopic
                    all_results.append(chunk)
                    seen_chunk_ids.add(chunk_id)

        logger.info(
            f"Retrieved {len(all_results)} unique chunks "
            f"across {len(subtopics)} subtopics"
        )
        return all_results

    def _enrich_with_document_info(self, results: list[dict]) -> list[dict]:
        """Add document titles and filenames to search results."""
        doc_ids = list({r["document_id"] for r in results if r.get("document_id")})

        doc_map = {}
        for doc_id in doc_ids:
            doc = self.db.fetch_one(
                f"SELECT id, title, filename FROM {TABLE_DOCUMENTS} WHERE id = ?",
                [doc_id],
            )
            if doc:
                doc_map[doc_id] = doc

        for result in results:
            doc_id = result.get("document_id", "")
            doc_info = doc_map.get(doc_id, {})
            result["document_title"] = doc_info.get("title", "Unknown")
            result["document_filename"] = doc_info.get("filename", "")

        return results

    def build_context_string(self, chunks: list[dict]) -> str:
        """
        Concatenate retrieved chunks into a context string
        suitable for LLM prompt injection.
        """
        sections = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("document_title", "Unknown source")
            content = chunk.get("content", "")
            score = chunk.get("similarity_score", 0)
            sections.append(
                f"[Source {i}: {source} (relevance: {score:.2f})]\n{content}"
            )

        return "\n\n---\n\n".join(sections)


# ── Module-level factory ─────────────────────────────────────

_content_retriever: Optional[ContentRetriever] = None


def get_content_retriever() -> ContentRetriever:
    global _content_retriever
    if _content_retriever is None:
        _content_retriever = ContentRetriever()
    return _content_retriever
