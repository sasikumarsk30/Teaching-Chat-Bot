"""
Metadata Manager

Tracks document processing metadata:
chunk counts, embedding status, processing timings.
"""

import logging
from typing import Optional

from app.core.constants import TABLE_DOCUMENTS, TABLE_CHUNKS
from app.infrastructure.data_access.duckdb_manager import get_duckdb_manager

logger = logging.getLogger(__name__)


class MetadataManager:
    """Manages document and chunk metadata in DuckDB."""

    def __init__(self):
        self.db = get_duckdb_manager()
        logger.info("MetadataManager initialized")

    def update_chunk_count(self, document_id: str, chunk_count: int) -> None:
        """Update the total_chunks field for a document."""
        self.db.execute(
            f"UPDATE {TABLE_DOCUMENTS} SET total_chunks = ? WHERE id = ?",
            [chunk_count, document_id],
        )
        logger.info(f"Updated chunk count | doc={document_id} chunks={chunk_count}")

    def mark_embeddings_generated(self, document_id: str) -> None:
        """Mark a document as having embeddings generated."""
        self.db.execute(
            f"UPDATE {TABLE_DOCUMENTS} SET embeddings_generated = TRUE WHERE id = ?",
            [document_id],
        )
        logger.info(f"Embeddings marked complete | doc={document_id}")

    def get_chunk_count(self, document_id: str) -> int:
        """Return the number of chunks for a document."""
        result = self.db.fetch_one(
            f"SELECT COUNT(*) as cnt FROM {TABLE_CHUNKS} WHERE document_id = ?",
            [document_id],
        )
        return result["cnt"] if result else 0

    def get_document_stats(self, document_id: str) -> Optional[dict]:
        """Return processing stats for a document."""
        doc = self.db.fetch_one(
            f"SELECT id, filename, title, total_chunks, embeddings_generated, "
            f"file_size_bytes, upload_date FROM {TABLE_DOCUMENTS} WHERE id = ?",
            [document_id],
        )
        if doc is None:
            return None

        chunk_count = self.get_chunk_count(document_id)
        return {
            **doc,
            "actual_chunk_count": chunk_count,
        }

    def get_all_document_ids(self) -> list[str]:
        """Return all document IDs."""
        rows = self.db.fetch_all(f"SELECT id FROM {TABLE_DOCUMENTS}")
        return [r["id"] for r in rows]

    def delete_chunks_for_document(self, document_id: str) -> int:
        """Delete all chunks for a document from DuckDB."""
        result = self.db.fetch_one(
            f"SELECT COUNT(*) as cnt FROM {TABLE_CHUNKS} WHERE document_id = ?",
            [document_id],
        )
        count = result["cnt"] if result else 0

        self.db.execute(
            f"DELETE FROM {TABLE_CHUNKS} WHERE document_id = ?",
            [document_id],
        )
        logger.info(f"Deleted {count} chunks from DuckDB | doc={document_id}")
        return count


# ── Module-level factory ─────────────────────────────────────

_metadata_manager: Optional[MetadataManager] = None


def get_metadata_manager() -> MetadataManager:
    global _metadata_manager
    if _metadata_manager is None:
        _metadata_manager = MetadataManager()
    return _metadata_manager
