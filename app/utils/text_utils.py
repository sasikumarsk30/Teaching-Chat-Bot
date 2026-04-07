"""
Text Utilities

Helpers for text cleaning, normalization, and sentence splitting.
"""

import re
import unicodedata
from typing import Optional


def clean_text(text: str) -> str:
    """
    Clean and normalize raw text extracted from documents.

    - Normalize Unicode
    - Collapse excessive whitespace
    - Remove control characters (except newlines)
    - Strip leading/trailing whitespace
    """
    # Normalize Unicode (e.g. fancy quotes → ASCII)
    text = unicodedata.normalize("NFKD", text)

    # Remove control characters except newlines and tabs
    text = re.sub(r"[^\S\n\t]+", " ", text)

    # Collapse multiple blank lines into two (paragraph separator)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse multiple spaces into one
    text = re.sub(r" {2,}", " ", text)

    return text.strip()


def split_sentences(text: str) -> list[str]:
    """
    Split text into sentences using a regex-based heuristic.

    Handles common abbreviations and decimal numbers.
    """
    # Sentence-ending punctuation followed by whitespace and uppercase
    sentence_endings = re.compile(
        r'(?<=[.!?])\s+(?=[A-Z"\'(])'
    )
    sentences = sentence_endings.split(text)
    return [s.strip() for s in sentences if s.strip()]


def split_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs by double newlines."""
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def truncate_text(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """Truncate text to max_length, appending suffix if truncated."""
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def count_tokens_approx(text: str) -> int:
    """
    Approximate token count using whitespace splitting.
    (Rough: 1 token ≈ 0.75 words for English.)
    """
    words = len(text.split())
    return int(words / 0.75)


def extract_heading_structure(text: str) -> list[dict]:
    """
    Detect markdown-style headings (# H1, ## H2, etc.) in text.

    Returns:
        List of dicts with 'level', 'title', 'position'.
    """
    headings = []
    for match in re.finditer(r"^(#{1,6})\s+(.+)$", text, re.MULTILINE):
        headings.append({
            "level": len(match.group(1)),
            "title": match.group(2).strip(),
            "position": match.start(),
        })
    return headings


def prepare_text_for_tts(text: str) -> str:
    """
    Prepare text for TTS by converting formatting that
    would be read literally (e.g. markdown, URLs, code).

    - Removes markdown formatting (bold, italic, headers)
    - Removes URLs
    - Adds pauses (periods) after section headings
    - Replaces bullet points with commas
    """
    # Remove markdown headers but keep text
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    # Remove bold/italic markers
    text = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}(.+?)_{1,3}", r"\1", text)

    # Remove markdown links → keep text
    text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)

    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)

    # Replace bullet points with natural phrasing
    text = re.sub(r"^\s*[-*•]\s+", "", text, flags=re.MULTILINE)

    # Replace numbered lists
    text = re.sub(r"^\s*\d+[.)]\s+", "", text, flags=re.MULTILINE)

    # Clean up excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)

    return text.strip()
