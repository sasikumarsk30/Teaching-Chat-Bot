"""
Unit tests for app.services.content_analysis.prompt_processor

Tests query parsing, topic extraction, search term generation,
constraint detection, and mode validation.
"""

import pytest
from unittest.mock import patch, MagicMock
from app.services.content_analysis.prompt_processor import PromptProcessor


@pytest.fixture
def processor():
    """Fresh PromptProcessor instance."""
    with patch("app.services.content_analysis.prompt_processor.validate_query_mode") as mock_vm:
        mock_vm.side_effect = lambda m: m.lower().strip()
        return PromptProcessor()


# ── Topic Extraction ─────────────────────────────────────────

class TestTopicExtraction:
    def test_removes_explain_prefix(self, processor):
        result = processor.parse_query("explain machine learning", "explain")
        assert "machine learning" in result["topic"].lower()

    def test_removes_teach_me_about_prefix(self, processor):
        result = processor.parse_query("teach me about quantum computing", "teach")
        assert "quantum computing" in result["topic"].lower()

    def test_removes_what_is_prefix(self, processor):
        result = processor.parse_query("what is deep learning?", "explain")
        assert "deep learning" in result["topic"].lower()

    def test_removes_how_does_prefix(self, processor):
        result = processor.parse_query("how does backpropagation work?", "explain")
        assert "backpropagation" in result["topic"].lower()

    def test_plain_topic_preserved(self, processor):
        result = processor.parse_query("neural networks", "explain")
        assert "neural networks" in result["topic"].lower()

    def test_strips_question_mark(self, processor):
        result = processor.parse_query("what is Python?", "explain")
        assert "?" not in result["topic"]


# ── Search Term Generation ───────────────────────────────────

class TestSearchTermGeneration:
    def test_generates_search_terms(self, processor):
        result = processor.parse_query("explain supervised learning algorithms", "explain")
        terms = result["search_terms"]
        assert len(terms) > 0
        # Original query should be first term
        assert result["search_terms"][0] == "explain supervised learning algorithms"

    def test_removes_stop_words(self, processor):
        result = processor.parse_query("what is the role of attention in transformers", "explain")
        terms = result["search_terms"]
        # Stop words like "what", "is", "the", "of", "in" should be removed
        lower_terms = [t.lower() for t in terms[1:]]  # Skip the full query
        assert "the" not in lower_terms
        assert "of" not in lower_terms

    def test_includes_content_words(self, processor):
        result = processor.parse_query("explain gradient descent optimization", "explain")
        terms = result["search_terms"]
        term_str = " ".join(terms).lower()
        assert "gradient" in term_str
        assert "descent" in term_str
        assert "optimization" in term_str


# ── Constraint Detection ─────────────────────────────────────

class TestConstraintDetection:
    def test_detects_brief_constraint(self, processor):
        result = processor.parse_query("briefly explain neural networks", "explain")
        assert result["constraints"]["detail_level"] == "brief"

    def test_detects_detailed_constraint(self, processor):
        result = processor.parse_query("give me a comprehensive explanation of AI", "explain")
        assert result["constraints"]["detail_level"] == "detailed"

    def test_default_standard_constraint(self, processor):
        result = processor.parse_query("explain machine learning", "explain")
        assert result["constraints"]["detail_level"] == "standard"

    def test_detects_summary_keyword(self, processor):
        result = processor.parse_query("give me a summary of deep learning", "explain")
        assert result["constraints"]["detail_level"] == "brief"


# ── Full Parse Output ────────────────────────────────────────

class TestParseQuery:
    def test_output_structure(self, processor):
        result = processor.parse_query("explain neural networks", "explain")
        assert "original_query" in result
        assert "topic" in result
        assert "mode" in result
        assert "search_terms" in result
        assert "constraints" in result

    def test_preserves_original_query(self, processor):
        query = "How do transformers work in NLP?"
        result = processor.parse_query(query, "teach")
        assert result["original_query"] == query

    def test_mode_passed_through(self, processor):
        result = processor.parse_query("explain AI", "teach")
        assert result["mode"] == "teach"
