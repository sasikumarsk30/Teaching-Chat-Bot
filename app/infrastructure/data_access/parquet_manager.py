"""
Parquet Manager

Handles reading and writing Parquet files for chunks and embeddings.
Uses PyArrow for efficient columnar storage.
"""

import logging
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

from app.core.config import get_settings, CHUNKS_DIR, EMBEDDINGS_DIR
from app.core.constants import (
    PARQUET_CHUNKS_METADATA,
    PARQUET_CHUNKS_VECTORS,
    PARQUET_DOCUMENTS_INDEX,
)

logger = logging.getLogger(__name__)


class ParquetManager:
    """Manages Parquet file I/O for chunks and embeddings."""

    def __init__(self):
        self.settings = get_settings()
        self.chunks_metadata_path = CHUNKS_DIR / PARQUET_CHUNKS_METADATA
        self.chunks_vectors_path = CHUNKS_DIR / PARQUET_CHUNKS_VECTORS
        self.documents_index_path = CHUNKS_DIR / PARQUET_DOCUMENTS_INDEX
        logger.info("ParquetManager initialized")

    # ── Write Operations ─────────────────────────────────────

    def write_chunks_metadata(self, df: pd.DataFrame) -> None:
        """Write or append chunk metadata to Parquet."""
        self._write_or_append(df, self.chunks_metadata_path)
        logger.info(
            f"Wrote {len(df)} chunk records to {self.chunks_metadata_path.name}"
        )

    def write_chunk_vectors(self, df: pd.DataFrame) -> None:
        """Write or append chunk embeddings to Parquet."""
        self._write_or_append(df, self.chunks_vectors_path)
        logger.info(
            f"Wrote {len(df)} vector records to {self.chunks_vectors_path.name}"
        )

    def write_documents_index(self, df: pd.DataFrame) -> None:
        """Write or append document index to Parquet."""
        self._write_or_append(df, self.documents_index_path)
        logger.info(
            f"Wrote {len(df)} document records to {self.documents_index_path.name}"
        )

    # ── Read Operations ──────────────────────────────────────

    def read_chunks_metadata(
        self,
        document_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """Read chunk metadata, optionally filtered by document."""
        if not self.chunks_metadata_path.exists():
            return pd.DataFrame()

        df = pq.read_table(self.chunks_metadata_path).to_pandas()
        if document_id:
            df = df[df["document_id"] == document_id]
        return df

    def read_chunk_vectors(
        self,
        chunk_ids: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Read chunk vectors, optionally filtered by chunk IDs."""
        if not self.chunks_vectors_path.exists():
            return pd.DataFrame()

        df = pq.read_table(self.chunks_vectors_path).to_pandas()
        if chunk_ids:
            df = df[df["id"].isin(chunk_ids)]
        return df

    def read_all_vectors_with_content(self) -> pd.DataFrame:
        meta_df = self.read_chunks_metadata()
        vec_df = self.read_chunk_vectors()

        if meta_df.empty or vec_df.empty:
            return pd.DataFrame()

        merged = meta_df.merge( 
            vec_df[["chunk_id", "embedding"]],
            left_on="id",
            right_on="chunk_id",
            how="inner",
        )
        return merged

    def read_documents_index(self) -> pd.DataFrame:
        """Read the documents index."""
        if not self.documents_index_path.exists():
            return pd.DataFrame()
        return pq.read_table(self.documents_index_path).to_pandas()

    # ── Delete Operations ────────────────────────────────────

    def delete_document_chunks(self, document_id: str) -> int:
        """Remove all chunks and vectors for a given document."""
        deleted = 0

        # Remove from chunks metadata
        if self.chunks_metadata_path.exists():
            df = pq.read_table(self.chunks_metadata_path).to_pandas()
            chunk_ids = df[df["document_id"] == document_id]["id"].tolist()
            original_count = len(df)
            df = df[df["document_id"] != document_id]
            deleted = original_count - len(df)

            if df.empty:
                self.chunks_metadata_path.unlink(missing_ok=True)
            else:
                table = pa.Table.from_pandas(df)
                pq.write_table(table, self.chunks_metadata_path)

            # Remove corresponding vectors
            if self.chunks_vectors_path.exists() and chunk_ids:
                vec_df = pq.read_table(self.chunks_vectors_path).to_pandas()
                vec_df = vec_df[~vec_df["id"].isin(chunk_ids)]
                if vec_df.empty:
                    self.chunks_vectors_path.unlink(missing_ok=True)
                else:
                    table = pa.Table.from_pandas(vec_df)
                    pq.write_table(table, self.chunks_vectors_path)

        # Remove from documents index
        if self.documents_index_path.exists():
            idx_df = pq.read_table(self.documents_index_path).to_pandas()
            idx_df = idx_df[idx_df["id"] != document_id]
            if idx_df.empty:
                self.documents_index_path.unlink(missing_ok=True)
            else:
                table = pa.Table.from_pandas(idx_df)
                pq.write_table(table, self.documents_index_path)

        logger.info(
            f"Deleted {deleted} chunks for document {document_id}"
        )
        return deleted

    # ── Internal Helpers ─────────────────────────────────────

    def _write_or_append(self, df: pd.DataFrame, path: Path) -> None:
        """Write a new Parquet file or append to an existing one."""
        if path.exists():
            existing = pq.read_table(path).to_pandas()
            combined = pd.concat([existing, df], ignore_index=True)
            # De-duplicate by 'id' column if present
            if "id" in combined.columns:
                combined = combined.drop_duplicates(subset=["id"], keep="last")
            table = pa.Table.from_pandas(combined)
        else:
            table = pa.Table.from_pandas(df)

        pq.write_table(table, path)


# ── Module-level factory ─────────────────────────────────────

_parquet_manager: Optional[ParquetManager] = None


def get_parquet_manager() -> ParquetManager:
    """Return a shared ParquetManager instance."""
    global _parquet_manager
    if _parquet_manager is None:
        _parquet_manager = ParquetManager()
    return _parquet_manager
