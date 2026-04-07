"""
Vector Store Service

Stores and manages chunk embeddings in Parquet files.
Bridges the embedding service and the parquet layer.
"""

import logging
import pandas as pd
from typing import Optional

from app.infrastructure.data_access.parquet_manager import get_parquet_manager
from app.infrastructure.data_access.duckdb_manager import get_duckdb_manager
from app.core.constants import TABLE_CHUNK_VECTORS

logger = logging.getLogger(__name__)


class VectorStoreService:
    """Persists and retrieves embedding vectors from Parquet + DuckDB."""

    def __init__(self):
        self.parquet = get_parquet_manager()
        self.db = get_duckdb_manager()
        logger.info("VectorStoreService initialized")

    def store_vectors(self, vector_records: list[dict]) -> int:
        """
        Persist vector records to both Parquet and DuckDB.

        Args:
            vector_records: List of dicts with id, chunk_id, embedding, etc.

        Returns:
            Number of vectors stored.
        """
        if not vector_records:
            return 0

        df = pd.DataFrame(vector_records)
        self.parquet.write_chunk_vectors(df)

        # Also insert into DuckDB for SQL-based search
        for rec in vector_records:
            self.db.execute(
                f"""
                INSERT OR REPLACE INTO {TABLE_CHUNK_VECTORS}
                    (id, chunk_id, embedding, vector_dim, model_name, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    rec["id"],
                    rec["chunk_id"],
                    rec["embedding"],
                    rec["vector_dim"],
                    rec.get("model_name", ""),
                    rec["created_at"],
                ],
            )

        count = len(vector_records)
        logger.info(f"Stored {count} vectors in Parquet + DuckDB")
        return count

    def get_vectors_for_document(self, document_id: str) -> pd.DataFrame:
        """Retrieve all vectors for chunks belonging to a document."""
        return self.parquet.read_all_vectors_with_content()

    def get_all_vectors(self) -> pd.DataFrame:
        """Retrieve all stored vectors with their chunk content."""
        return self.parquet.read_all_vectors_with_content()

    def delete_vectors_for_document(self, document_id: str) -> int:
        """Remove vectors associated with a document."""
        deleted = self.parquet.delete_document_chunks(document_id)
        logger.info(f"Deleted vectors for document {document_id}")
        return deleted

    def get_vector_count(self) -> int:
        """Return total number of stored vectors."""
        return self.db.row_count(TABLE_CHUNK_VECTORS)


# ── Module-level factory ─────────────────────────────────────

_vector_store: Optional[VectorStoreService] = None


def get_vector_store_service() -> VectorStoreService:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStoreService()
    return _vector_store
