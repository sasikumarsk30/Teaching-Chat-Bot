"""
Unit tests for app.utils.file_utils

Tests file validation, extension detection, MIME types,
filename sanitization, and human-readable sizes.
"""

import pytest
from unittest.mock import patch, MagicMock
from app.utils.file_utils import (
    get_file_extension,
    get_mime_type,
    sanitize_filename,
    get_file_size_human,
    validate_file,
)
from app.utils.error_handlers import UnsupportedFileTypeError, FileTooLargeError


# ── get_file_extension ───────────────────────────────────────

class TestGetFileExtension:
    def test_pdf(self):
        assert get_file_extension("report.pdf") == ".pdf"

    def test_docx(self):
        assert get_file_extension("document.docx") == ".docx"

    def test_txt(self):
        assert get_file_extension("notes.txt") == ".txt"

    def test_md(self):
        assert get_file_extension("README.md") == ".md"

    def test_uppercase_extension(self):
        assert get_file_extension("FILE.PDF") == ".pdf"

    def test_multiple_dots(self):
        assert get_file_extension("my.document.v2.pdf") == ".pdf"

    def test_no_extension(self):
        assert get_file_extension("noext") == ""


# ── get_mime_type ────────────────────────────────────────────

class TestGetMimeType:
    def test_pdf_mime(self):
        assert get_mime_type("file.pdf") == "application/pdf"

    def test_txt_mime(self):
        assert get_mime_type("file.txt") == "text/plain"

    def test_md_mime(self):
        assert get_mime_type("file.md") == "text/markdown"

    def test_docx_mime(self):
        mime = get_mime_type("file.docx")
        assert "wordprocessingml" in mime

    def test_unknown_mime(self):
        assert get_mime_type("file.xyz") == "application/octet-stream"


# ── sanitize_filename ────────────────────────────────────────

class TestSanitizeFilename:
    def test_preserves_safe_characters(self):
        assert sanitize_filename("my_document-v2.pdf") == "my_document-v2.pdf"

    def test_replaces_special_chars(self):
        result = sanitize_filename("file<with>bad|chars.txt")
        assert "<" not in result
        assert ">" not in result
        assert "|" not in result
        assert result.endswith(".txt")

    def test_preserves_extension(self):
        result = sanitize_filename("test!!!.pdf")
        assert result.endswith(".pdf")

    def test_preserves_spaces(self):
        result = sanitize_filename("my file name.docx")
        assert " " in result
        assert result.endswith(".docx")


# ── get_file_size_human ─────────────────────────────────────

class TestGetFileSizeHuman:
    def test_bytes(self):
        result = get_file_size_human(500)
        assert "B" in result

    def test_kilobytes(self):
        result = get_file_size_human(2048)
        assert "KB" in result

    def test_megabytes(self):
        result = get_file_size_human(5 * 1024 * 1024)
        assert "MB" in result

    def test_gigabytes(self):
        result = get_file_size_human(2 * 1024 * 1024 * 1024)
        assert "GB" in result

    def test_zero(self):
        result = get_file_size_human(0)
        assert "B" in result


# ── validate_file ────────────────────────────────────────────

class TestValidateFile:
    def test_valid_pdf(self):
        ext = validate_file("report.pdf", 1024)
        assert ext == ".pdf"

    def test_valid_txt(self):
        ext = validate_file("notes.txt", 500)
        assert ext == ".txt"

    def test_valid_docx(self):
        ext = validate_file("doc.docx", 2048)
        assert ext == ".docx"

    def test_valid_md(self):
        ext = validate_file("readme.md", 256)
        assert ext == ".md"

    def test_unsupported_file_type_raises(self):
        with pytest.raises(UnsupportedFileTypeError):
            validate_file("image.jpg", 1024)

    def test_file_too_large_raises(self):
        # Default max is 50 MB
        huge_size = 100 * 1024 * 1024  # 100 MB
        with pytest.raises(FileTooLargeError):
            validate_file("big.pdf", huge_size)

    def test_exact_max_size_passes(self):
        # 50 MB exactly should pass
        size = 50 * 1024 * 1024
        ext = validate_file("file.pdf", size)
        assert ext == ".pdf"
