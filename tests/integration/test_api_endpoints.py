"""
Integration tests for FastAPI API endpoints using TestClient.

Tests the HTTP layer: request validation, response format,
status codes, and error handling. Uses mocked service dependencies
to avoid requiring real models or external services.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime
from fastapi.testclient import TestClient

from app.main import create_app


# ── App fixture with mocked services ─────────────────────────

@pytest.fixture
def app():
    """Create a FastAPI app with mocked lifespan and services."""
    with patch("app.main.get_duckdb_manager") as mock_db, \
         patch("app.main.get_staging_manager") as mock_staging:
        db = MagicMock()
        db.connection = True
        db.initialize_schema = MagicMock()
        mock_db.return_value = db

        staging = MagicMock()
        staging.shutdown = MagicMock()
        mock_staging.return_value = staging

        application = create_app()
        yield application


@pytest.fixture
def client(app):
    """TestClient for the FastAPI app."""
    return TestClient(app)


# ═══════════════════════════════════════════════════════════════
# Health & Root Endpoints
# ═══════════════════════════════════════════════════════════════

class TestHealthEndpoints:
    def test_root_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "service" in data
        assert "version" in data
        assert data["docs"] == "/docs"

    def test_health_returns_200(self, client):
        with patch("app.main.get_duckdb_manager") as mock_db:
            db = MagicMock()
            db.connection = True
            mock_db.return_value = db

            resp = client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "healthy"
            assert "version" in data
            assert "services" in data


# ═══════════════════════════════════════════════════════════════
# Document Endpoints
# ═══════════════════════════════════════════════════════════════

class TestDocumentEndpoints:
    def test_list_documents_empty(self, client):
        with patch("app.endpoints.document_endpoints.get_document_ingestion_service") as mock_svc:
            svc = MagicMock()
            svc.list_documents.return_value = ([], 0)
            mock_svc.return_value = svc

            resp = client.get("/api/v1/documents")
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is True
            assert data["documents"] == []
            assert data["total_count"] == 0

    def test_get_document_not_found(self, client):
        with patch("app.endpoints.document_endpoints.get_document_ingestion_service") as mock_svc:
            svc = MagicMock()
            svc.get_document.return_value = None
            mock_svc.return_value = svc

            resp = client.get("/api/v1/documents/nonexistent-id")
            assert resp.status_code == 404

    def test_get_document_found(self, client):
        with patch("app.endpoints.document_endpoints.get_document_ingestion_service") as mock_svc:
            svc = MagicMock()
            svc.get_document.return_value = {
                "id": "doc-123",
                "filename": "test.txt",
                "title": "Test",
                "file_type": ".txt",
                "file_size_bytes": 1024,
                "total_chunks": 5,
                "embeddings_generated": True,
                "tags": [],
                "upload_date": datetime.utcnow(),
            }
            mock_svc.return_value = svc

            resp = client.get("/api/v1/documents/doc-123")
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is True
            assert data["document"]["id"] == "doc-123"

    def test_delete_document_not_found(self, client):
        with patch("app.endpoints.document_endpoints.get_document_ingestion_service") as mock_svc, \
             patch("app.endpoints.document_endpoints.get_metadata_manager"), \
             patch("app.endpoints.document_endpoints.get_parquet_manager"):
            svc = MagicMock()
            svc.get_document.return_value = None
            mock_svc.return_value = svc

            resp = client.delete("/api/v1/documents/nonexistent-id")
            assert resp.status_code == 404


# ═══════════════════════════════════════════════════════════════
# Chunk Endpoints
# ═══════════════════════════════════════════════════════════════

class TestChunkEndpoints:
    def test_list_chunks_empty(self, client):
        with patch("app.endpoints.chunk_endpoints.get_duckdb_manager") as mock_db:
            db = MagicMock()
            db.fetch_all.return_value = []
            db.fetch_one.return_value = {"cnt": 0}
            mock_db.return_value = db

            resp = client.get("/api/v1/chunks")
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is True
            assert data["chunks"] == []

    def test_get_chunk_not_found(self, client):
        with patch("app.endpoints.chunk_endpoints.get_duckdb_manager") as mock_db:
            db = MagicMock()
            db.fetch_one.return_value = None
            mock_db.return_value = db

            resp = client.get("/api/v1/chunks/nonexistent")
            assert resp.status_code == 404

    def test_get_chunk_found(self, client):
        with patch("app.endpoints.chunk_endpoints.get_duckdb_manager") as mock_db:
            db = MagicMock()
            db.fetch_one.return_value = {
                "id": "chunk-001",
                "document_id": "doc-001",
                "sequence": 0,
                "content": "Test chunk content",
                "chunk_size": 18,
                "start_char": 0,
                "end_char": 18,
                "metadata": None,
            }
            mock_db.return_value = db

            resp = client.get("/api/v1/chunks/chunk-001")
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is True
            assert data["chunk"]["id"] == "chunk-001"


# ═══════════════════════════════════════════════════════════════
# Query Endpoints
# ═══════════════════════════════════════════════════════════════

class TestQueryEndpoints:
    def test_query_explain(self, client):
        with patch("app.endpoints.query_endpoints.get_response_generator") as mock_gen, \
             patch("app.endpoints.query_endpoints.validate_query_mode") as mock_vm:

            mock_vm.return_value = "explain"
            gen = AsyncMock()
            gen.generate = AsyncMock(return_value={
                "response_id": "resp-001",
                "response_text": "Machine learning is...",
                "source_chunks": [],
                "mode": "explain",
                "from_cache": False,
                "processing_time_ms": 150.0,
            })
            mock_gen.return_value = gen

            resp = client.post("/api/v1/query", json={
                "question": "What is machine learning?",
                "mode": "explain",
            })
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is True
            assert data["response_text"] == "Machine learning is..."
            assert data["mode"] == "explain"

    def test_query_teach(self, client):
        with patch("app.endpoints.query_endpoints.get_response_generator") as mock_gen, \
             patch("app.endpoints.query_endpoints.validate_query_mode") as mock_vm:

            mock_vm.return_value = "teach"
            gen = AsyncMock()
            gen.generate = AsyncMock(return_value={
                "response_id": "resp-002",
                "response_text": "Lesson on ML...",
                "source_chunks": [],
                "mode": "teach",
                "from_cache": False,
                "processing_time_ms": 200.0,
            })
            mock_gen.return_value = gen

            resp = client.post("/api/v1/query/teach", json={
                "question": "Teach me about neural networks",
                "mode": "teach",
            })
            assert resp.status_code == 200

    def test_query_validation_rejects_short_question(self, client):
        resp = client.post("/api/v1/query", json={
            "question": "ab",  # Too short (min_length=3)
            "mode": "explain",
        })
        assert resp.status_code == 422  # Validation error


# ═══════════════════════════════════════════════════════════════
# Audio Endpoints
# ═══════════════════════════════════════════════════════════════

class TestAudioEndpoints:
    def test_list_audio_empty(self, client):
        with patch("app.endpoints.audio_endpoints.get_tts_service") as mock_tts:
            tts = MagicMock()
            tts.list_audio_files.return_value = []
            mock_tts.return_value = tts

            resp = client.get("/api/v1/audio")
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is True
            assert data["audio_files"] == []

    def test_download_audio_not_found(self, client):
        with patch("app.endpoints.audio_endpoints.get_tts_service") as mock_tts:
            tts = MagicMock()
            tts.get_audio_file.return_value = None
            mock_tts.return_value = tts

            resp = client.get("/api/v1/audio/nonexistent/download")
            assert resp.status_code == 404

    def test_delete_audio_not_found(self, client):
        with patch("app.endpoints.audio_endpoints.get_tts_service") as mock_tts:
            tts = MagicMock()
            tts.delete_audio.return_value = False
            mock_tts.return_value = tts

            resp = client.delete("/api/v1/audio/nonexistent")
            assert resp.status_code == 404

    def test_generate_audio_from_missing_response(self, client):
        with patch("app.endpoints.audio_endpoints.get_response_cache") as mock_cache, \
             patch("app.endpoints.audio_endpoints.get_tts_service"):
            cache = MagicMock()
            cache.get_by_id.return_value = None
            mock_cache.return_value = cache

            resp = client.post("/api/v1/audio/generate", json={
                "response_id": "nonexistent",
                "output_format": "mp3",
            })
            assert resp.status_code == 404
