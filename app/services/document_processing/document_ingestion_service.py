"""
Document Ingestion Service

Handles uploading, parsing, and validating documents.
Extracts text content from PDF, DOCX, TXT, and Markdown files.
"""

import logging
from pathlib import Path
from typing import Optional

from app.core.config import get_settings
from app.infrastructure.data_access.document_store import get_document_store
from app.utils.file_utils import validate_file, get_file_extension
from app.utils.text_utils import clean_text

logger = logging.getLogger(__name__)


class DocumentIngestionService:
    """Parses and ingests uploaded documents."""

    def __init__(self):
        self.settings = get_settings()
        self.document_store = get_document_store()
        logger.info("DocumentIngestionService initialized")

    async def ingest_document(
        self,
        file_content: bytes,
        filename: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> dict:
        """
        Validate, parse, store, and return metadata for an uploaded document.

        Returns:
            dict with document metadata and extracted text content.
        """
        # 1. Validate file
        file_ext = validate_file(filename, len(file_content))
        doc_title = title or Path(filename).stem

        logger.info(f"Ingesting document: {filename} ({file_ext})")

        # 2. Extract text content
        raw_text = self._extract_text(file_content, file_ext)
        text_content = clean_text(raw_text)

        if not text_content:
            logger.warning(f"No text extracted from {filename}")

        # 3. Store document
        doc_metadata = self.document_store.save_document(
            file_content=file_content,
            filename=filename,
            title=doc_title,
            description=description,
            file_type=file_ext,
            tags=tags,
        )

        doc_metadata["text_content"] = text_content
        doc_metadata["character_count"] = len(text_content)

        logger.info(
            f"Document ingested | id={doc_metadata['id']} "
            f"chars={len(text_content)}"
        )

        return doc_metadata

    def _extract_text(self, content: bytes, file_ext: str) -> str:
        """Route to the appropriate text extraction method."""
        extractors = {
            ".pdf": self._extract_from_pdf,
            ".docx": self._extract_from_docx,
            ".txt": self._extract_from_text,
            ".md": self._extract_from_text,
        }

        extractor = extractors.get(file_ext)
        if extractor is None:
            logger.error(f"No extractor for file type: {file_ext}")
            return ""

        try:
            return extractor(content)
        except Exception as e:
            logger.error(f"Text extraction failed ({file_ext}): {e}")
            return ""

    def _extract_from_pdf(self, content: bytes) -> str:
        """Extract text from a PDF file using PyPDF2."""
        try:
            import io
            from PyPDF2 import PdfReader

            reader = PdfReader(io.BytesIO(content))
            pages = []
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    pages.append(page_text)

            text = "\n\n".join(pages)
            logger.info(f"PDF extracted | pages={len(reader.pages)} chars={len(text)}")
            return text

        except ImportError:
            logger.error("PyPDF2 not installed. Run: pip install PyPDF2")
            raise
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            raise

    def _extract_from_docx(self, content: bytes) -> str:
        """Extract text from a DOCX file using python-docx."""
        try:
            import io
            from docx import Document

            doc = Document(io.BytesIO(content))
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
            text = "\n\n".join(paragraphs)
            logger.info(f"DOCX extracted | paragraphs={len(paragraphs)} chars={len(text)}")
            return text

        except ImportError:
            logger.error("python-docx not installed. Run: pip install python-docx")
            raise
        except Exception as e:
            logger.error(f"DOCX extraction error: {e}")
            raise

    def _extract_from_text(self, content: bytes) -> str:
        """Extract text from plain text / markdown files."""
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            text = content.decode("latin-1")
        logger.info(f"Text file extracted | chars={len(text)}")
        return text

    def get_document(self, doc_id: str) -> Optional[dict]:
        """Retrieve document metadata."""
        return self.document_store.get_document(doc_id)

    def list_documents(self, page: int = 1, page_size: int = 20):
        """List documents with pagination."""
        return self.document_store.list_documents(page, page_size)

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and its stored file."""
        return self.document_store.delete_document(doc_id)


# ── Module-level factory ─────────────────────────────────────

_ingestion_service: Optional[DocumentIngestionService] = None


def get_document_ingestion_service() -> DocumentIngestionService:
    global _ingestion_service
    if _ingestion_service is None:
        _ingestion_service = DocumentIngestionService()
    return _ingestion_service
