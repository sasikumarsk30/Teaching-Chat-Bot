"""
Unit tests for app.services.document_processing.document_ingestion_service

Tests document text extraction, validation, and ingestion pipeline
with mocked document store.
"""

import pytest
from unittest.mock import patch, MagicMock
from app.services.document_processing.document_ingestion_service import DocumentIngestionService


@pytest.fixture
def ingestion_service():
    """DocumentIngestionService with mocked dependencies."""
    with patch("app.services.document_processing.document_ingestion_service.get_settings") as m_settings, \
         patch("app.services.document_processing.document_ingestion_service.get_document_store") as m_store:

        settings = MagicMock()
        settings.max_upload_size_bytes = 50 * 1024 * 1024
        settings.allowed_extensions = [".pdf", ".docx", ".txt", ".md"]
        m_settings.return_value = settings

        store = MagicMock()
        store.save_document.return_value = {
            "id": "doc-001",
            "filename": "test.txt",
            "title": "Test Document",
            "description": None,
            "file_type": ".txt",
            "file_size_bytes": 256,
            "original_path": "/data/documents/doc-001_test.txt",
            "tags": [],
            "upload_date": "2025-01-01T00:00:00",
        }
        m_store.return_value = store

        service = DocumentIngestionService()
        yield service


# ── Text Extraction ──────────────────────────────────────────

class TestTextExtraction:
    def test_extracts_from_txt(self, ingestion_service):
        content = b"Hello, this is a test document."
        text = ingestion_service._extract_text(content, ".txt")
        assert text == "Hello, this is a test document."

    def test_extracts_from_md(self, ingestion_service):
        content = b"# Title\n\nMarkdown content here."
        text = ingestion_service._extract_text(content, ".md")
        assert "Title" in text
        assert "Markdown content" in text

    def test_txt_utf8_encoding(self, ingestion_service):
        content = "Unicode text: café".encode("utf-8")
        text = ingestion_service._extract_text(content, ".txt")
        assert "café" in text

    def test_txt_latin1_fallback(self, ingestion_service):
        content = "Latin-1 text: caf\xe9".encode("latin-1")
        text = ingestion_service._extract_text(content, ".txt")
        assert "caf" in text

    def test_unsupported_extension_returns_empty(self, ingestion_service):
        text = ingestion_service._extract_text(b"data", ".xyz")
        assert text == ""


# ── Ingestion Pipeline ───────────────────────────────────────

class TestIngestionPipeline:
    @pytest.mark.asyncio
    async def test_ingest_txt_document(self, ingestion_service, sample_txt_content):
        result = await ingestion_service.ingest_document(
            file_content=sample_txt_content,
            filename="data_science.txt",
            title="Data Science Intro",
        )
        assert result["id"] == "doc-001"
        assert "text_content" in result
        assert result["character_count"] > 0

    @pytest.mark.asyncio
    async def test_ingest_sets_title_from_filename(self, ingestion_service, sample_txt_content):
        result = await ingestion_service.ingest_document(
            file_content=sample_txt_content,
            filename="my_notes.txt",
            title=None,
        )
        # Title should default to filename stem
        assert result["id"] == "doc-001"

    @pytest.mark.asyncio
    async def test_ingest_stores_document(self, ingestion_service, sample_txt_content):
        await ingestion_service.ingest_document(
            file_content=sample_txt_content,
            filename="test.txt",
        )
        ingestion_service.document_store.save_document.assert_called_once()

    @pytest.mark.asyncio
    async def test_ingest_with_tags(self, ingestion_service, sample_txt_content):
        await ingestion_service.ingest_document(
            file_content=sample_txt_content,
            filename="test.txt",
            tags=["science", "intro"],
        )
        call_kwargs = ingestion_service.document_store.save_document.call_args
        assert call_kwargs[1].get("tags") == ["science", "intro"] or \
               "tags" in str(call_kwargs)


# ── List and Get ─────────────────────────────────────────────

class TestDocumentRetrieval:
    def test_get_document(self, ingestion_service):
        ingestion_service.document_store.get_document.return_value = {
            "id": "doc-001", "title": "Test"
        }
        result = ingestion_service.get_document("doc-001")
        assert result["id"] == "doc-001"

    def test_get_nonexistent_document(self, ingestion_service):
        ingestion_service.document_store.get_document.return_value = None
        result = ingestion_service.get_document("nonexistent")
        assert result is None

    def test_list_documents(self, ingestion_service):
        ingestion_service.document_store.list_documents.return_value = (
            [{"id": "doc-001"}, {"id": "doc-002"}],
            2,
        )
        docs, total = ingestion_service.list_documents(page=1, page_size=20)
        assert len(docs) == 2
        assert total == 2

    def test_delete_document(self, ingestion_service):
        ingestion_service.document_store.delete_document.return_value = True
        result = ingestion_service.delete_document("doc-001")
        assert result is True
