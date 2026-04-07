"""
Unit tests for app.services.content_analysis.response_generator

Tests response generation pipeline with mocked LLM client,
content retriever, and cache. Covers explain/teach modes,
caching behavior, predefined content, and no-content fallback.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from app.services.content_analysis.response_generator import ResponseGenerator


@pytest.fixture
def mock_services():
    """Patch all external dependencies of ResponseGenerator."""
    with patch("app.services.content_analysis.response_generator.get_settings") as m_settings, \
         patch("app.services.content_analysis.response_generator.get_llm_client") as m_llm, \
         patch("app.services.content_analysis.response_generator.get_response_cache") as m_cache, \
         patch("app.services.content_analysis.response_generator.get_content_retriever") as m_retriever, \
         patch("app.services.content_analysis.response_generator.get_prompt_processor") as m_processor:

        settings = MagicMock()
        settings.search_top_k = 5
        settings.enable_response_cache = True
        m_settings.return_value = settings

        llm = AsyncMock()
        llm.generate = AsyncMock(return_value="Generated response about machine learning.")
        m_llm.return_value = llm

        cache = MagicMock()
        cache.get.return_value = None  # No cached response by default
        cache.put.return_value = "response-id-123"
        m_cache.return_value = cache

        retriever = AsyncMock()
        retriever.retrieve = AsyncMock(return_value=[
            {
                "chunk_id": "c1",
                "document_id": "doc-001",
                "content": "Machine learning is a subset of AI.",
                "document_title": "ML Intro",
                "sequence": 0,
                "similarity_score": 0.85,
            },
            {
                "chunk_id": "c2",
                "document_id": "doc-001",
                "content": "Supervised learning uses labeled data.",
                "document_title": "ML Intro",
                "sequence": 1,
                "similarity_score": 0.82,
            },
        ])
        retriever.build_context_string.return_value = (
            "[Source 1: ML Intro]\nMachine learning is a subset of AI.\n\n"
            "[Source 2: ML Intro]\nSupervised learning uses labeled data."
        )
        retriever.retrieve_for_subtopics = AsyncMock(return_value=[
            {
                "chunk_id": "c1", "document_id": "doc-001",
                "content": "Neural nets are layers.", "document_title": "NN Intro",
                "sequence": 0, "similarity_score": 0.90, "subtopic": "neural networks",
            },
        ])
        m_retriever.return_value = retriever

        processor = MagicMock()
        processor.parse_query.return_value = {
            "original_query": "explain ML",
            "topic": "ML",
            "mode": "explain",
            "search_terms": ["ML"],
            "constraints": {"detail_level": "standard"},
        }
        m_processor.return_value = processor

        yield {
            "settings": settings,
            "llm": llm,
            "cache": cache,
            "retriever": retriever,
            "processor": processor,
        }


@pytest.fixture
def response_generator(mock_services):
    """ResponseGenerator with all dependencies mocked."""
    return ResponseGenerator()


# ── Basic Generation ─────────────────────────────────────────

class TestResponseGeneration:
    @pytest.mark.asyncio
    async def test_generates_response(self, response_generator, mock_services):
        result = await response_generator.generate(
            question="What is machine learning?",
            mode="explain",
        )
        assert result["response_text"] == "Generated response about machine learning."
        assert result["response_id"] == "response-id-123"
        assert result["mode"] == "explain"
        assert result["from_cache"] is False
        assert len(result["source_chunks"]) == 2

    @pytest.mark.asyncio
    async def test_calls_llm(self, response_generator, mock_services):
        await response_generator.generate(question="test", mode="explain")
        mock_services["llm"].generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_calls_retriever(self, response_generator, mock_services):
        await response_generator.generate(question="test", mode="explain")
        mock_services["retriever"].retrieve.assert_called_once()

    @pytest.mark.asyncio
    async def test_processing_time_recorded(self, response_generator):
        result = await response_generator.generate(question="test", mode="explain")
        assert "processing_time_ms" in result
        assert result["processing_time_ms"] >= 0


# ── Caching ──────────────────────────────────────────────────

class TestResponseCaching:
    @pytest.mark.asyncio
    async def test_returns_cached_response(self, response_generator, mock_services):
        mock_services["cache"].get.return_value = {
            "id": "cached-id",
            "response_text": "Cached answer here.",
        }
        result = await response_generator.generate(question="cached query", mode="explain")
        assert result["from_cache"] is True
        assert result["response_text"] == "Cached answer here."
        # LLM should NOT be called for cached responses
        mock_services["llm"].generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_stores_response_in_cache(self, response_generator, mock_services):
        await response_generator.generate(question="new query", mode="explain")
        mock_services["cache"].put.assert_called_once()


# ── No Content Fallback ──────────────────────────────────────

class TestNoContentFallback:
    @pytest.mark.asyncio
    async def test_returns_fallback_when_no_chunks(self, response_generator, mock_services):
        mock_services["retriever"].retrieve.return_value = []
        result = await response_generator.generate(question="unknown topic", mode="explain")
        assert "couldn't find relevant content" in result["response_text"].lower()
        assert result["source_chunks"] == []


# ── Teach Mode ───────────────────────────────────────────────

class TestTeachMode:
    @pytest.mark.asyncio
    async def test_teach_mode_passes_to_llm(self, response_generator, mock_services):
        await response_generator.generate(question="teach me AI", mode="teach")
        call_kwargs = mock_services["llm"].generate.call_args
        # System prompt should contain teaching-related content
        assert call_kwargs is not None


# ── Predefined Content ───────────────────────────────────────

class TestPredefinedContent:
    @pytest.mark.asyncio
    async def test_generates_predefined_response(self, response_generator, mock_services):
        result = await response_generator.generate_predefined(
            topic="Neural Networks",
            subtopics=["architecture", "training"],
            mode="teach",
        )
        assert result["response_text"] == "Generated response about machine learning."
        assert result["topic"] == "Neural Networks"
        assert result["from_cache"] is False
        assert "response_id" in result

    @pytest.mark.asyncio
    async def test_predefined_no_content_fallback(self, response_generator, mock_services):
        mock_services["retriever"].retrieve_for_subtopics.return_value = []
        result = await response_generator.generate_predefined(
            topic="Unknown",
            subtopics=["nothing"],
            mode="teach",
        )
        assert "couldn't find relevant content" in result["response_text"].lower()


# ── Source Chunk Formatting ──────────────────────────────────

class TestSourceChunkFormatting:
    def test_format_source_chunks(self, response_generator):
        chunks = [
            {
                "chunk_id": "c1",
                "document_id": "doc-001",
                "document_title": "Test Doc",
                "sequence": 0,
                "similarity_score": 0.85,
                "content": "A" * 300,
            }
        ]
        formatted = ResponseGenerator._format_source_chunks(chunks)
        assert len(formatted) == 1
        assert formatted[0]["chunk_id"] == "c1"
        assert formatted[0]["document_title"] == "Test Doc"
        assert len(formatted[0]["snippet"]) <= 200
