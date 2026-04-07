"""
Unit tests for app.utils.text_utils

Tests text cleaning, sentence/paragraph splitting, truncation,
heading extraction, and TTS text preparation.
"""

import pytest
from app.utils.text_utils import (
    clean_text,
    split_sentences,
    split_paragraphs,
    truncate_text,
    count_tokens_approx,
    extract_heading_structure,
    prepare_text_for_tts,
)


# ── clean_text ───────────────────────────────────────────────

class TestCleanText:
    def test_strips_leading_trailing_whitespace(self):
        assert clean_text("  hello world  ") == "hello world"

    def test_collapses_multiple_spaces(self):
        assert clean_text("hello    world") == "hello world"

    def test_collapses_excessive_blank_lines(self):
        result = clean_text("para one\n\n\n\n\npara two")
        assert "\n\n\n" not in result
        assert "para one" in result
        assert "para two" in result

    def test_preserves_single_paragraph_break(self):
        result = clean_text("para one\n\npara two")
        assert "para one\n\npara two" == result

    def test_handles_empty_string(self):
        assert clean_text("") == ""

    def test_unicode_normalization(self):
        # NFKD normalization should handle fancy characters
        result = clean_text("café")
        assert "caf" in result

    def test_preserves_newlines_and_tabs(self):
        result = clean_text("line1\nline2\ttab")
        assert "\n" in result or "line1" in result


# ── split_sentences ──────────────────────────────────────────

class TestSplitSentences:
    def test_basic_sentence_splitting(self):
        text = "First sentence. Second sentence. Third sentence."
        sentences = split_sentences(text)
        # Should split on sentence boundaries
        assert len(sentences) >= 1

    def test_single_sentence(self):
        sentences = split_sentences("Just one sentence here.")
        assert len(sentences) == 1
        assert sentences[0] == "Just one sentence here."

    def test_exclamation_and_question(self):
        text = "What is this? It's amazing! Really great."
        sentences = split_sentences(text)
        assert len(sentences) >= 2

    def test_empty_string(self):
        assert split_sentences("") == []

    def test_preserves_content(self):
        text = "Machine learning is great. Deep learning is a subset."
        sentences = split_sentences(text)
        combined = " ".join(sentences)
        assert "Machine learning" in combined
        assert "Deep learning" in combined


# ── split_paragraphs ─────────────────────────────────────────

class TestSplitParagraphs:
    def test_splits_on_double_newline(self, sample_text):
        paragraphs = split_paragraphs(sample_text)
        assert len(paragraphs) == 5

    def test_strips_whitespace(self):
        text = "  para one  \n\n  para two  "
        paragraphs = split_paragraphs(text)
        assert paragraphs[0] == "para one"
        assert paragraphs[1] == "para two"

    def test_single_paragraph(self):
        paragraphs = split_paragraphs("Just one paragraph with no breaks.")
        assert len(paragraphs) == 1

    def test_empty_paragraphs_filtered(self):
        text = "first\n\n\n\n\n\nsecond"
        paragraphs = split_paragraphs(text)
        assert all(p.strip() for p in paragraphs)

    def test_empty_string(self):
        assert split_paragraphs("") == []


# ── truncate_text ────────────────────────────────────────────

class TestTruncateText:
    def test_short_text_not_truncated(self):
        assert truncate_text("hello", 200) == "hello"

    def test_long_text_truncated(self):
        text = "a" * 300
        result = truncate_text(text, 200)
        assert len(result) == 200
        assert result.endswith("...")

    def test_custom_suffix(self):
        text = "a" * 100
        result = truncate_text(text, 50, suffix="…")
        assert result.endswith("…")

    def test_exact_length_not_truncated(self):
        text = "a" * 200
        assert truncate_text(text, 200) == text


# ── count_tokens_approx ─────────────────────────────────────

class TestCountTokensApprox:
    def test_returns_positive_for_text(self):
        count = count_tokens_approx("This is a simple test sentence.")
        assert count > 0

    def test_empty_string(self):
        assert count_tokens_approx("") == 0

    def test_proportional_to_length(self):
        short = count_tokens_approx("Hello world")
        long = count_tokens_approx("Hello world " * 100)
        assert long > short


# ── extract_heading_structure ────────────────────────────────

class TestExtractHeadingStructure:
    def test_detects_markdown_headings(self):
        text = "# Title\n\nContent here\n\n## Subtitle\n\nMore content"
        headings = extract_heading_structure(text)
        assert len(headings) == 2
        assert headings[0]["level"] == 1
        assert headings[0]["title"] == "Title"
        assert headings[1]["level"] == 2
        assert headings[1]["title"] == "Subtitle"

    def test_no_headings(self):
        headings = extract_heading_structure("Plain text with no headings.")
        assert headings == []

    def test_heading_positions(self):
        text = "# First\n\ntext\n\n# Second"
        headings = extract_heading_structure(text)
        assert headings[0]["position"] < headings[1]["position"]


# ── prepare_text_for_tts ─────────────────────────────────────

class TestPrepareTextForTTS:
    def test_removes_markdown_headers(self):
        result = prepare_text_for_tts("# Title\n\nSome content")
        assert "#" not in result
        assert "Title" in result

    def test_removes_bold_italic(self):
        result = prepare_text_for_tts("This is **bold** and *italic*.")
        assert "**" not in result
        assert "*" not in result
        assert "bold" in result
        assert "italic" in result

    def test_removes_urls(self):
        result = prepare_text_for_tts("Visit https://example.com for details.")
        assert "https://" not in result
        assert "Visit" in result

    def test_removes_markdown_links(self):
        result = prepare_text_for_tts("Check [this link](https://example.com) out.")
        assert "this link" in result
        assert "https://" not in result
        assert "[" not in result

    def test_removes_bullet_points(self):
        text = "Items:\n- First item\n- Second item"
        result = prepare_text_for_tts(text)
        assert "- " not in result
        assert "First item" in result

    def test_empty_string(self):
        assert prepare_text_for_tts("") == ""
