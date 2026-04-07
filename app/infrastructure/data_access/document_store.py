"""
Document Store

Handles persistence of uploaded document files on the local filesystem
and tracks metadata in DuckDB.
"""

import logging
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.core.config import DOCUMENTS_DIR
from app.core.constants import TABLE_DOCUMENTS
from app.infrastructure.data_access.duckdb_manager import get_duckdb_manager

logger = logging.getLogger(__name__)


class DocumentStore:
    """Manages document file storage and metadata persistence."""

    def __init__(self):
        self.storage_dir = DOCUMENTS_DIR
        self.db = get_duckdb_manager()
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"DocumentStore initialized | dir={self.storage_dir}")

    def save_document(
        self,
        file_content: bytes,
        filename: str,
        title: str,
        description: Optional[str] = None,
        file_type: str = "",
        tags: Optional[list[str]] = None,
    ) -> dict:
        """
        Save a document to disk and register it in DuckDB.

        Returns:
            dict with document metadata including generated 'id'.
        """
        doc_id = str(uuid.uuid4())
        safe_filename = f"{doc_id}_{filename}"
        file_path = self.storage_dir / safe_filename

        # Write file to disk
        file_path.write_bytes(file_content)
        file_size = len(file_content)

        # Insert metadata into DuckDB
        self.db.execute(
            f"""
            INSERT INTO {TABLE_DOCUMENTS}
                (id, filename, title, description, file_type,
                 file_size_bytes, original_path, tags, upload_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                doc_id,
                filename,
                title,
                description,
                file_type,
                file_size,
                str(file_path),
                tags or [],
                datetime.utcnow(),
            ],
        )

        logger.info(f"Document saved | id={doc_id} filename={filename}")

        return {
            "id": doc_id,
            "filename": filename,
            "title": title,
            "description": description,
            "file_type": file_type,
            "file_size_bytes": file_size,
            "original_path": str(file_path),
            "tags": tags or [],
            "upload_date": datetime.utcnow(),
            "total_chunks": 0,
            "embeddings_generated": False,
        }

    def get_document(self, doc_id: str) -> Optional[dict]:
        """Retrieve document metadata by ID."""
        return self.db.fetch_one(
            f"SELECT * FROM {TABLE_DOCUMENTS} WHERE id = ?", [doc_id]
        )

    def list_documents(
        self, page: int = 1, page_size: int = 20
    ) -> tuple[list[dict], int]:
        """Return paginated list of documents and total count."""
        total = self.db.row_count(TABLE_DOCUMENTS)
        offset = (page - 1) * page_size
        docs = self.db.fetch_all(
            f"SELECT * FROM {TABLE_DOCUMENTS} "
            f"ORDER BY upload_date DESC LIMIT ? OFFSET ?",
            [page_size, offset],
        )
        return docs, total

    def update_document(self, doc_id: str, **kwargs) -> bool:
        """Update specific fields of a document."""
        if not kwargs:
            return False

        set_clauses = ", ".join(f"{k} = ?" for k in kwargs)
        values = list(kwargs.values()) + [doc_id]
        self.db.execute(
            f"UPDATE {TABLE_DOCUMENTS} SET {set_clauses} WHERE id = ?",
            values,
        )
        logger.info(f"Document updated | id={doc_id} fields={list(kwargs.keys())}")
        return True

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document file and its metadata."""
        doc = self.get_document(doc_id)
        if doc is None:
            logger.warning(f"Document not found for deletion | id={doc_id}")
            return False

        # Remove file from disk
        file_path = Path(doc["original_path"])
        if file_path.exists():
            file_path.unlink()

        # Remove from DuckDB (cascade will be handled by callers for chunks)
        self.db.execute(
            f"DELETE FROM {TABLE_DOCUMENTS} WHERE id = ?", [doc_id]
        )
        logger.info(f"Document deleted | id={doc_id}")
        return True

    def read_document_content(self, doc_id: str) -> Optional[bytes]:
        """Read raw file content from disk."""
        doc = self.get_document(doc_id)
        if doc is None:
            return None

        file_path = Path(doc["original_path"])
        if not file_path.exists():
            logger.error(f"Document file missing on disk | id={doc_id}")
            return None

        return file_path.read_bytes()


# ── Module-level factory ─────────────────────────────────────

_document_store: Optional[DocumentStore] = None


def get_document_store() -> DocumentStore:
    global _document_store
    if _document_store is None:
        _document_store = DocumentStore()
    return _document_store
