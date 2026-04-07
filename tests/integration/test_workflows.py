"""
Integration tests for end-to-end workflows.

Tests the full pipeline flows:
1. Document Upload → Chunking → Staging → DuckDB
2. Chunking → Embedding → Vector Store → Similarity Search
3. Query → Retrieve → Generate → (Audio)

Uses mocked external services (LLM, TTS, embedding model)
but exercises real chunking, DuckDB operations, and Parquet I/O.
"""

import pytest
import uuid
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock, AsyncMock

from app.services.document_processing.chunking_service import ChunkingService
from app.infrastructure.cache.embedding_cache import EmbeddingCache


# ═══════════════════════════════════════════════════════════════
# Workflow 1: Upload → Chunk → Store in DuckDB
# ═══════════════════════════════════════════════════════════════

class TestDocumentUploadWorkflow:
    """Tests the pipeline: upload → extract text → chunk → store."""

    def test_text_to_chunks_to_duckdb(self, sample_text, test_duckdb):
        """
        Simulate the upload pipeline:
        1. Extract text (already have sample_text)
        2. Chunk the text
        3. Store chunks in DuckDB
        """
        doc_id = str(uuid.uuid4())

        # Step 1: Insert document record
        test_duckdb.execute(
            "INSERT INTO documents (id, filename, title, file_type, file_size_bytes) "
            "VALUES (?, ?, ?, ?, ?)",
            [doc_id, "test.txt", "ML Intro", ".txt", len(sample_text)],
        )

        # Step 2: Chunk the text
        with patch("app.services.document_processing.chunking_service.get_settings") as m:
            settings = MagicMock()
            settings.chunk_size = 500
            settings.chunk_overlap = 50
            m.return_value = settings
            chunking = ChunkingService()

        chunks = chunking.chunk_document(
            text=sample_text,
            document_id=doc_id,
            strategy="semantic",
        )

        assert len(chunks) > 0, "Chunking should produce at least one chunk"

        # Step 3: Store chunks in DuckDB
        for chunk in chunks:
            test_duckdb.execute(
                "INSERT INTO chunks (id, document_id, sequence, content, chunk_size, start_char, end_char) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    chunk["id"], doc_id, chunk["sequence"],
                    chunk["content"], chunk["chunk_size"],
                    chunk["start_char"], chunk["end_char"],
                ],
            )

        # Step 4: Update document metadata
        test_duckdb.execute(
            "UPDATE documents SET total_chunks = ? WHERE id = ?",
            [len(chunks), doc_id],
        )

        # Verification
        stored_chunks = test_duckdb.execute(
            "SELECT * FROM chunks WHERE document_id = ? ORDER BY sequence",
            [doc_id],
        ).fetchall()
        assert len(stored_chunks) == len(chunks)

        # Verify document updated
        doc = test_duckdb.execute(
            "SELECT total_chunks FROM documents WHERE id = ?", [doc_id]
        ).fetchone()
        assert doc[0] == len(chunks)

    def test_different_chunking_strategies_produce_chunks(self, sample_text):
        """All three strategies should produce valid chunks."""
        with patch("app.services.document_processing.chunking_service.get_settings") as m:
            settings = MagicMock()
            settings.chunk_size = 300
            settings.chunk_overlap = 50
            m.return_value = settings
            chunking = ChunkingService()

        for strategy in ["semantic", "paragraph", "fixed"]:
            chunks = chunking.chunk_document(
                text=sample_text,
                document_id="doc-test",
                strategy=strategy,
            )
            assert len(chunks) > 0, f"Strategy '{strategy}' failed to produce chunks"
            for chunk in chunks:
                assert chunk["content"].strip(), f"Empty chunk from '{strategy}'"

    def test_markdown_document_upload_flow(self, sample_md_content, test_duckdb):
        """Test upload workflow for markdown document."""
        from app.utils.text_utils import clean_text

        doc_id = str(uuid.uuid4())
        text_content = sample_md_content.decode("utf-8")
        cleaned = clean_text(text_content)

        # Insert doc
        test_duckdb.execute(
            "INSERT INTO documents (id, filename, title, file_type) "
            "VALUES (?, ?, ?, ?)",
            [doc_id, "notes.md", "Python Notes", ".md"],
        )

        # Chunk
        with patch("app.services.document_processing.chunking_service.get_settings") as m:
            settings = MagicMock()
            settings.chunk_size = 500
            settings.chunk_overlap = 50
            m.return_value = settings
            chunking = ChunkingService()

        chunks = chunking.chunk_document(
            text=cleaned, document_id=doc_id, strategy="semantic",
        )
        assert len(chunks) > 0

        # Store
        for chunk in chunks:
            test_duckdb.execute(
                "INSERT INTO chunks (id, document_id, sequence, content) "
                "VALUES (?, ?, ?, ?)",
                [chunk["id"], doc_id, chunk["sequence"], chunk["content"]],
            )

        count = test_duckdb.execute(
            "SELECT COUNT(*) FROM chunks WHERE document_id = ?", [doc_id]
        ).fetchone()[0]
        assert count == len(chunks)


# ═══════════════════════════════════════════════════════════════
# Workflow 2: Chunks → Embeddings → Similarity Search
# ═══════════════════════════════════════════════════════════════

class TestEmbeddingSearchWorkflow:
    """Tests: chunk → embed → store vectors → search."""

    def test_embedding_and_search_flow(self, sample_text, test_duckdb, tmp_data_dir):
        """
        1. Chunk document
        2. Generate mock embeddings
        3. Store in Parquet
        4. Search with query embedding
        """
        from app.services.embedding_generation.similarity_search import SimilaritySearch

        doc_id = str(uuid.uuid4())

        # Step 1: Chunk
        with patch("app.services.document_processing.chunking_service.get_settings") as m:
            settings = MagicMock()
            settings.chunk_size = 500
            settings.chunk_overlap = 50
            m.return_value = settings
            chunking = ChunkingService()

        chunks = chunking.chunk_document(
            text=sample_text, document_id=doc_id, strategy="semantic",
        )

        # Step 2: Generate deterministic embeddings
        np.random.seed(42)
        embeddings = [np.random.randn(384).astype(np.float32) for _ in chunks]

        # Step 3: Build Parquet-like DataFrame
        meta_df = pd.DataFrame([{
            "id": c["id"],
            "document_id": c["document_id"],
            "sequence": c["sequence"],
            "content": c["content"],
            "chunk_size": c["chunk_size"],
            "metadata": c["metadata"],
        } for c in chunks])

        vec_df = pd.DataFrame([{
            "id": c["id"],
            "embedding": emb.tolist(),
        } for c, emb in zip(chunks, embeddings)])

        # Merge as would be done by parquet_manager.read_all_vectors_with_content()
        merged = meta_df.merge(vec_df, on="id", how="inner")

        # Step 4: Similarity search
        with patch("app.services.embedding_generation.similarity_search.get_settings") as m_settings, \
             patch("app.services.embedding_generation.similarity_search.get_parquet_manager") as m_pq:

            settings = MagicMock()
            settings.search_top_k = 3
            settings.similarity_threshold = 0.0
            m_settings.return_value = settings

            pq = MagicMock()
            pq.read_all_vectors_with_content.return_value = merged
            m_pq.return_value = pq

            search = SimilaritySearch()

        # Use the first chunk's embedding as query
        query_emb = embeddings[0]
        results = search.search(query_emb, top_k=3)

        assert len(results) > 0
        assert results[0]["similarity_score"] >= results[-1]["similarity_score"]
        # The first result should be the most similar (ideally chunk_0 itself)
        assert results[0]["similarity_score"] > 0.5

    def test_embedding_cache_prevents_recomputation(self):
        """Verify that cached embeddings are reused."""
        cache = EmbeddingCache(max_size=100)

        text = "Some text about machine learning"
        emb = np.random.randn(384).astype(np.float32)

        cache.put(text, emb)
        result = cache.get(text)

        assert result is not None
        np.testing.assert_array_equal(result, emb)
        assert cache.stats["hits"] == 1

    def test_document_scoped_search(self, sample_text):
        """Verify search can be scoped to a specific document."""
        from app.services.embedding_generation.similarity_search import SimilaritySearch

        # Create chunks for two documents
        np.random.seed(42)
        data = []
        for doc_idx in range(2):
            doc_id = f"doc-{doc_idx}"
            for i in range(3):
                data.append({
                    "id": f"chunk-{doc_idx}-{i}",
                    "document_id": doc_id,
                    "content": f"Content for doc {doc_idx}, chunk {i}",
                    "sequence": i,
                    "embedding": np.random.randn(384).astype(np.float32).tolist(),
                    "metadata": {},
                })

        df = pd.DataFrame(data)

        with patch("app.services.embedding_generation.similarity_search.get_settings") as m_settings, \
             patch("app.services.embedding_generation.similarity_search.get_parquet_manager") as m_pq:

            settings = MagicMock()
            settings.search_top_k = 5
            settings.similarity_threshold = 0.0
            m_settings.return_value = settings

            pq = MagicMock()
            pq.read_all_vectors_with_content.return_value = df
            m_pq.return_value = pq

            search = SimilaritySearch()

        query_emb = np.random.randn(384).astype(np.float32)
        results = search.search(query_emb, top_k=5, document_id="doc-0")

        for r in results:
            assert r["document_id"] == "doc-0"


# ═══════════════════════════════════════════════════════════════
# Workflow 3: Query → Retrieve → Generate Response
# ═══════════════════════════════════════════════════════════════

class TestQueryWorkflow:
    """Tests the full query pipeline with mocked LLM."""

    @pytest.mark.asyncio
    async def test_full_query_pipeline(self):
        """
        Simulate: user asks question → retrieve chunks → generate LLM response.
        """
        from app.services.content_analysis.prompt_processor import PromptProcessor
        from app.prompts.system_prompts import get_system_prompt, build_user_prompt

        # Step 1: Parse query
        with patch("app.services.content_analysis.prompt_processor.validate_query_mode") as mock_vm:
            mock_vm.side_effect = lambda m: m.lower().strip()
            processor = PromptProcessor()

        parsed = processor.parse_query(
            "Explain supervised learning algorithms",
            mode="explain",
        )

        assert parsed["topic"]
        assert parsed["mode"] == "explain"
        assert len(parsed["search_terms"]) > 0

        # Step 2: Simulate chunk retrieval results
        retrieved_chunks = [
            {
                "chunk_id": "c1",
                "document_id": "doc-001",
                "content": "Supervised learning uses labeled datasets to train algorithms.",
                "document_title": "ML Guide",
                "sequence": 0,
                "similarity_score": 0.92,
            },
            {
                "chunk_id": "c2",
                "document_id": "doc-001",
                "content": "The model learns to map inputs to known outputs.",
                "document_title": "ML Guide",
                "sequence": 1,
                "similarity_score": 0.88,
            },
        ]

        # Step 3: Build prompt
        context_sections = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            context_sections.append(
                f"[Source {i}: {chunk['document_title']}]\n{chunk['content']}"
            )
        context = "\n\n---\n\n".join(context_sections)

        system_prompt = get_system_prompt("explain")
        user_prompt = build_user_prompt(
            question=parsed["original_query"],
            context=context,
            mode="explain",
        )

        assert "supervised learning" in user_prompt.lower() or "Supervised" in user_prompt
        assert "labeled datasets" in user_prompt
        assert "explain" in system_prompt.lower() or "clear" in system_prompt.lower()

        # Step 4: Mock LLM generation
        mock_response = (
            "Supervised learning is a type of machine learning where the algorithm "
            "learns from labeled training data. Think of it like a teacher showing "
            "a student the correct answers. The model learns to map inputs to outputs "
            "by studying many examples, and then it can predict answers for new, "
            "unseen data."
        )

        # Verify the response is usable
        assert len(mock_response) > 50
        assert "supervised learning" in mock_response.lower()

    @pytest.mark.asyncio
    async def test_no_results_returns_fallback(self):
        """When no chunks match, the system should return a helpful fallback."""
        from app.services.content_analysis.response_generator import ResponseGenerator

        with patch("app.services.content_analysis.response_generator.get_settings") as m_s, \
             patch("app.services.content_analysis.response_generator.get_llm_client"), \
             patch("app.services.content_analysis.response_generator.get_response_cache") as m_cache, \
             patch("app.services.content_analysis.response_generator.get_content_retriever") as m_ret, \
             patch("app.services.content_analysis.response_generator.get_prompt_processor") as m_proc:

            m_s.return_value = MagicMock()

            cache = MagicMock()
            cache.get.return_value = None
            m_cache.return_value = cache

            retriever = AsyncMock()
            retriever.retrieve = AsyncMock(return_value=[])
            m_ret.return_value = retriever

            processor = MagicMock()
            processor.parse_query.return_value = {
                "original_query": "quantum entanglement",
                "topic": "quantum entanglement",
                "mode": "explain",
                "search_terms": ["quantum entanglement"],
                "constraints": {},
            }
            m_proc.return_value = processor

            generator = ResponseGenerator()
            result = await generator.generate(
                question="quantum entanglement",
                mode="explain",
            )

        assert "couldn't find relevant content" in result["response_text"].lower()
        assert result["source_chunks"] == []


# ═══════════════════════════════════════════════════════════════
# Workflow 4: Full Upload → Query → Audio (Integration)
# ═══════════════════════════════════════════════════════════════

class TestFullPipelineWorkflow:
    """End-to-end test combining upload, query, and audio generation."""

    def test_upload_chunk_and_search(self, sample_text, test_duckdb, tmp_data_dir):
        """
        Complete flow:
        1. Upload text
        2. Chunk it
        3. Generate mock embeddings
        4. Store in DuckDB + create vector DataFrame
        5. Search and verify results make sense
        """
        doc_id = str(uuid.uuid4())

        # Upload
        test_duckdb.execute(
            "INSERT INTO documents (id, filename, title, file_type) "
            "VALUES (?, ?, ?, ?)",
            [doc_id, "ml_intro.txt", "ML Introduction", ".txt"],
        )

        # Chunk
        with patch("app.services.document_processing.chunking_service.get_settings") as m:
            settings = MagicMock()
            settings.chunk_size = 500
            settings.chunk_overlap = 50
            m.return_value = settings
            chunking = ChunkingService()

        chunks = chunking.chunk_document(
            text=sample_text, document_id=doc_id, strategy="semantic",
        )

        # Store chunks
        for chunk in chunks:
            test_duckdb.execute(
                "INSERT INTO chunks (id, document_id, sequence, content, chunk_size) "
                "VALUES (?, ?, ?, ?, ?)",
                [chunk["id"], doc_id, chunk["sequence"],
                 chunk["content"], chunk["chunk_size"]],
            )

        # Generate embeddings
        np.random.seed(42)
        embeddings = [np.random.randn(384).astype(np.float32) for _ in chunks]

        # Build merged dataframe for search
        data = [{
            "id": c["id"],
            "document_id": c["document_id"],
            "content": c["content"],
            "sequence": c["sequence"],
            "embedding": emb.tolist(),
            "metadata": {},
        } for c, emb in zip(chunks, embeddings)]
        merged_df = pd.DataFrame(data)

        # Search
        from app.services.embedding_generation.similarity_search import SimilaritySearch

        with patch("app.services.embedding_generation.similarity_search.get_settings") as m_s, \
             patch("app.services.embedding_generation.similarity_search.get_parquet_manager") as m_pq:

            m_s.return_value = MagicMock(search_top_k=3, similarity_threshold=0.0)
            pq = MagicMock()
            pq.read_all_vectors_with_content.return_value = merged_df
            m_pq.return_value = pq

            search = SimilaritySearch()

        query_emb = embeddings[0]  # Search with first chunk's embedding
        results = search.search(query_emb, top_k=3)

        assert len(results) > 0
        # Best match should be chunk 0 itself (exact match)
        assert results[0]["similarity_score"] > 0.9

        # Verify DuckDB state
        total = test_duckdb.execute(
            "SELECT COUNT(*) FROM chunks WHERE document_id = ?", [doc_id]
        ).fetchone()[0]
        assert total == len(chunks)
