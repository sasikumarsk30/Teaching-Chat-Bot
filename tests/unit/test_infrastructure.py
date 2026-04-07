"""
Unit tests for infrastructure components:
- DuckDB Manager (schema, CRUD)
- Staging Manager (buffer, flush)
- Response Cache (TTL, lookup)
- System Prompts (template building)
"""

import pytest
import uuid
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import duckdb


# ═══════════════════════════════════════════════════════════════
# DuckDB Schema & Query Tests (using test_duckdb fixture)
# ═══════════════════════════════════════════════════════════════

class TestDuckDBSchema:
    def test_tables_created(self, test_duckdb):
        """Verify all tables exist after schema initialization."""
        result = test_duckdb.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main'"
        ).fetchall()
        table_names = {r[0] for r in result}
        assert "documents" in table_names
        assert "chunks" in table_names
        assert "chunk_vectors" in table_names
        assert "response_cache" in table_names

    def test_insert_and_query_document(self, test_duckdb):
        doc_id = str(uuid.uuid4())
        test_duckdb.execute(
            "INSERT INTO documents (id, filename, title, file_type, file_size_bytes) "
            "VALUES (?, ?, ?, ?, ?)",
            [doc_id, "test.txt", "Test Doc", ".txt", 1024],
        )
        row = test_duckdb.execute(
            "SELECT * FROM documents WHERE id = ?", [doc_id]
        ).fetchone()
        assert row is not None
        assert row[1] == "test.txt"  # filename

    def test_insert_and_query_chunk(self, test_duckdb):
        doc_id = str(uuid.uuid4())
        chunk_id = str(uuid.uuid4())
        test_duckdb.execute(
            "INSERT INTO documents (id, filename, title, file_type) "
            "VALUES (?, ?, ?, ?)",
            [doc_id, "test.txt", "Test", ".txt"],
        )
        test_duckdb.execute(
            "INSERT INTO chunks (id, document_id, sequence, content, chunk_size) "
            "VALUES (?, ?, ?, ?, ?)",
            [chunk_id, doc_id, 0, "Chunk content here.", 19],
        )
        row = test_duckdb.execute(
            "SELECT content FROM chunks WHERE id = ?", [chunk_id]
        ).fetchone()
        assert row[0] == "Chunk content here."

    def test_count_chunks_for_document(self, test_duckdb):
        doc_id = str(uuid.uuid4())
        test_duckdb.execute(
            "INSERT INTO documents (id, filename, title, file_type) "
            "VALUES (?, ?, ?, ?)",
            [doc_id, "test.txt", "Test", ".txt"],
        )
        for i in range(5):
            test_duckdb.execute(
                "INSERT INTO chunks (id, document_id, sequence, content) "
                "VALUES (?, ?, ?, ?)",
                [str(uuid.uuid4()), doc_id, i, f"Content {i}"],
            )
        count = test_duckdb.execute(
            "SELECT COUNT(*) FROM chunks WHERE document_id = ?", [doc_id]
        ).fetchone()[0]
        assert count == 5

    def test_delete_cascades_manually(self, test_duckdb):
        """Test manual cascade delete (chunks then document)."""
        doc_id = str(uuid.uuid4())
        test_duckdb.execute(
            "INSERT INTO documents (id, filename, title, file_type) "
            "VALUES (?, ?, ?, ?)",
            [doc_id, "test.txt", "Test", ".txt"],
        )
        test_duckdb.execute(
            "INSERT INTO chunks (id, document_id, sequence, content) "
            "VALUES (?, ?, ?, ?)",
            [str(uuid.uuid4()), doc_id, 0, "Chunk content"],
        )
        # Delete chunks first
        test_duckdb.execute(
            "DELETE FROM chunks WHERE document_id = ?", [doc_id]
        )
        # Then delete document
        test_duckdb.execute(
            "DELETE FROM documents WHERE id = ?", [doc_id]
        )
        doc = test_duckdb.execute(
            "SELECT * FROM documents WHERE id = ?", [doc_id]
        ).fetchone()
        assert doc is None


# ═══════════════════════════════════════════════════════════════
# Response Cache DuckDB Tests
# ═══════════════════════════════════════════════════════════════

class TestResponseCacheDuckDB:
    def test_insert_and_retrieve_cached_response(self, test_duckdb):
        entry_id = str(uuid.uuid4())
        expires = datetime.utcnow() + timedelta(hours=1)
        test_duckdb.execute(
            "INSERT INTO response_cache (id, query, mode, response_text, created_at, expires_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [entry_id, "what is ML?", "explain", "ML is great.", datetime.utcnow(), expires],
        )
        row = test_duckdb.execute(
            "SELECT * FROM response_cache WHERE id = ?", [entry_id]
        ).fetchone()
        assert row is not None

    def test_expired_entries_queryable(self, test_duckdb):
        entry_id = str(uuid.uuid4())
        expired = datetime.utcnow() - timedelta(hours=1)
        test_duckdb.execute(
            "INSERT INTO response_cache (id, query, mode, response_text, expires_at) "
            "VALUES (?, ?, ?, ?, ?)",
            [entry_id, "old query", "explain", "old answer", expired],
        )
        # Should still be in table but expired
        row = test_duckdb.execute(
            "SELECT * FROM response_cache WHERE id = ? AND expires_at > ?",
            [entry_id, datetime.utcnow()],
        ).fetchone()
        assert row is None  # Expired, so not returned with WHERE clause


# ═══════════════════════════════════════════════════════════════
# Staging Manager Tests (in-memory buffer logic)
# ═══════════════════════════════════════════════════════════════

class TestStagingManagerBuffer:
    def test_add_chunks_increments_buffer(self):
        """Test the staging buffer logic in isolation."""
        from app.infrastructure.data_access.staging_manager import StagingManager

        with patch("app.infrastructure.data_access.staging_manager.get_settings") as m_settings, \
             patch("app.infrastructure.data_access.staging_manager.get_duckdb_manager") as m_db, \
             patch("app.infrastructure.data_access.staging_manager.get_parquet_manager") as m_pq:

            m_settings.return_value = MagicMock()
            m_db.return_value = MagicMock()
            m_pq.return_value = MagicMock()

            staging = StagingManager(flush_threshold=100, flush_timeout_seconds=9999)

            chunks = [
                {"id": str(uuid.uuid4()), "document_id": "doc-001",
                 "sequence": i, "content": f"Chunk {i}"}
                for i in range(5)
            ]
            size = staging.add_chunks(chunks)
            assert size == 5
            assert staging.get_buffer_size() == 5

    def test_auto_flush_on_threshold(self):
        """Buffer should auto-flush when threshold is reached."""
        from app.infrastructure.data_access.staging_manager import StagingManager

        with patch("app.infrastructure.data_access.staging_manager.get_settings") as m_settings, \
             patch("app.infrastructure.data_access.staging_manager.get_duckdb_manager") as m_db, \
             patch("app.infrastructure.data_access.staging_manager.get_parquet_manager") as m_pq:

            m_settings.return_value = MagicMock()
            db = MagicMock()
            m_db.return_value = db
            pq = MagicMock()
            m_pq.return_value = pq

            staging = StagingManager(flush_threshold=5, flush_timeout_seconds=9999)

            chunks = [
                {"id": str(uuid.uuid4()), "document_id": "doc-001",
                 "sequence": i, "content": f"Chunk {i}"}
                for i in range(6)
            ]
            staging.add_chunks(chunks)
            # After adding 6 chunks with threshold=5, buffer should be flushed
            assert staging.get_buffer_size() == 0

    def test_clear_empties_buffer(self):
        from app.infrastructure.data_access.staging_manager import StagingManager

        with patch("app.infrastructure.data_access.staging_manager.get_settings") as m_settings, \
             patch("app.infrastructure.data_access.staging_manager.get_duckdb_manager") as m_db, \
             patch("app.infrastructure.data_access.staging_manager.get_parquet_manager") as m_pq:

            m_settings.return_value = MagicMock()
            m_db.return_value = MagicMock()
            m_pq.return_value = MagicMock()

            staging = StagingManager(flush_threshold=100)
            staging.add_chunks([
                {"id": "1", "document_id": "d", "sequence": 0, "content": "x"}
            ])
            staging.clear()
            assert staging.get_buffer_size() == 0

    def test_flush_serializes_metadata_as_valid_json(self):
        """Regression: metadata must be JSON-encoded (double quotes), not Python repr (single quotes)."""
        import json
        from app.infrastructure.data_access.staging_manager import StagingManager

        with patch("app.infrastructure.data_access.staging_manager.get_settings") as m_settings, \
             patch("app.infrastructure.data_access.staging_manager.get_duckdb_manager") as m_db, \
             patch("app.infrastructure.data_access.staging_manager.get_parquet_manager") as m_pq:

            m_settings.return_value = MagicMock()
            db_mock = MagicMock()
            m_db.return_value = db_mock
            pq_mock = MagicMock()
            m_pq.return_value = pq_mock

            staging = StagingManager(flush_threshold=1, flush_timeout_seconds=9999)
            staging.add_chunks([{
                "id": "chunk-1",
                "document_id": "doc-001",
                "sequence": 0,
                "content": "Test content",
                "metadata": {"strategy": "semantic", "original_chunk_size": 1000},
            }])

            # Verify db.execute was called
            assert db_mock.execute.called
            call_args = db_mock.execute.call_args[0]  # positional args
            params = call_args[1]  # SQL params list
            metadata_value = params[-1]  # metadata is the last param

            # Must be valid JSON (double quotes), not Python repr (single quotes)
            parsed = json.loads(metadata_value)
            assert parsed["strategy"] == "semantic"
            assert parsed["original_chunk_size"] == 1000
            # Confirm it's proper JSON, not Python repr
            assert "'" not in metadata_value


# ═══════════════════════════════════════════════════════════════
# System Prompts Tests
# ═══════════════════════════════════════════════════════════════

class TestSystemPrompts:
    def test_get_explain_prompt(self):
        from app.prompts.system_prompts import get_system_prompt, EXPLAIN_SYSTEM_PROMPT
        prompt = get_system_prompt("explain")
        assert prompt == EXPLAIN_SYSTEM_PROMPT
        assert "explain" in prompt.lower() or "clear" in prompt.lower()

    def test_get_teach_prompt(self):
        from app.prompts.system_prompts import get_system_prompt, TEACH_SYSTEM_PROMPT
        prompt = get_system_prompt("teach")
        assert prompt == TEACH_SYSTEM_PROMPT
        assert "teach" in prompt.lower() or "lesson" in prompt.lower()

    def test_unknown_mode_defaults_to_explain(self):
        from app.prompts.system_prompts import get_system_prompt, EXPLAIN_SYSTEM_PROMPT
        prompt = get_system_prompt("unknown")
        assert prompt == EXPLAIN_SYSTEM_PROMPT

    def test_build_user_prompt_includes_context(self):
        from app.prompts.system_prompts import build_user_prompt
        prompt = build_user_prompt(
            question="What is AI?",
            context="AI stands for artificial intelligence.",
            mode="explain",
        )
        assert "What is AI?" in prompt
        assert "artificial intelligence" in prompt

    def test_build_user_prompt_teach_mode(self):
        from app.prompts.system_prompts import build_user_prompt
        prompt = build_user_prompt(
            question="Explain deep learning",
            context="Deep learning uses neural networks.",
            mode="teach",
        )
        assert "Explain deep learning" in prompt
        assert "neural networks" in prompt
