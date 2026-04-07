"""
Unit tests for app.utils.validators and app.utils.error_handlers

Tests query mode validation, audio format validation,
custom exception classes, and error handler responses.
"""

import pytest
from app.utils.validators import validate_query_mode, validate_audio_format
from app.utils.error_handlers import (
    AppError,
    DocumentNotFoundError,
    ChunkNotFoundError,
    UnsupportedFileTypeError,
    FileTooLargeError,
    EmbeddingGenerationError,
    LLMGenerationError,
    TTSGenerationError,
    AudioNotFoundError,
    InvalidQueryModeError,
)
from app.core.constants import (
    ERROR_DOCUMENT_NOT_FOUND,
    ERROR_CHUNK_NOT_FOUND,
    ERROR_UNSUPPORTED_FILE_TYPE,
    ERROR_FILE_TOO_LARGE,
    ERROR_EMBEDDING_FAILED,
    ERROR_LLM_FAILED,
    ERROR_TTS_FAILED,
    ERROR_AUDIO_NOT_FOUND,
    ERROR_INVALID_QUERY_MODE,
)


# ── validate_query_mode ──────────────────────────────────────

class TestValidateQueryMode:
    def test_explain_mode(self):
        assert validate_query_mode("explain") == "explain"

    def test_teach_mode(self):
        assert validate_query_mode("teach") == "teach"

    def test_case_insensitive(self):
        assert validate_query_mode("EXPLAIN") == "explain"
        assert validate_query_mode("Teach") == "teach"

    def test_strips_whitespace(self):
        assert validate_query_mode("  explain  ") == "explain"

    def test_invalid_mode_raises(self):
        with pytest.raises(InvalidQueryModeError):
            validate_query_mode("summarize")

    def test_empty_string_raises(self):
        with pytest.raises(InvalidQueryModeError):
            validate_query_mode("")


# ── validate_audio_format ────────────────────────────────────

class TestValidateAudioFormat:
    def test_mp3(self):
        assert validate_audio_format("mp3") == "mp3"

    def test_wav(self):
        assert validate_audio_format("wav") == "wav"

    def test_ogg(self):
        assert validate_audio_format("ogg") == "ogg"

    def test_case_insensitive(self):
        assert validate_audio_format("MP3") == "mp3"

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError) as exc_info:
            validate_audio_format("flac")
        assert "Unsupported audio format" in str(exc_info.value)


# ── Custom Exception Classes ─────────────────────────────────

class TestAppError:
    def test_base_app_error(self):
        err = AppError("Test error", "TEST_CODE", 400)
        assert err.message == "Test error"
        assert err.error_code == "TEST_CODE"
        assert err.status_code == 400
        assert str(err) == "Test error"

    def test_default_status_code(self):
        err = AppError("Server error")
        assert err.status_code == 500
        assert err.error_code == "INTERNAL_ERROR"


class TestDocumentNotFoundError:
    def test_message_includes_id(self):
        err = DocumentNotFoundError("doc-123")
        assert "doc-123" in err.message
        assert err.status_code == 404
        assert err.error_code == ERROR_DOCUMENT_NOT_FOUND


class TestChunkNotFoundError:
    def test_message_includes_id(self):
        err = ChunkNotFoundError("chunk-456")
        assert "chunk-456" in err.message
        assert err.status_code == 404
        assert err.error_code == ERROR_CHUNK_NOT_FOUND


class TestUnsupportedFileTypeError:
    def test_message_includes_type_and_allowed(self):
        err = UnsupportedFileTypeError(".exe", [".pdf", ".txt"])
        assert ".exe" in err.message
        assert ".pdf" in err.message
        assert err.status_code == 400
        assert err.error_code == ERROR_UNSUPPORTED_FILE_TYPE


class TestFileTooLargeError:
    def test_message_includes_sizes(self):
        err = FileTooLargeError(100 * 1024 * 1024, 50 * 1024 * 1024)
        assert "100.0 MB" in err.message or "100" in err.message
        assert err.status_code == 413
        assert err.error_code == ERROR_FILE_TOO_LARGE


class TestEmbeddingGenerationError:
    def test_with_detail(self):
        err = EmbeddingGenerationError("Model not found")
        assert "Model not found" in err.message
        assert err.status_code == 500
        assert err.error_code == ERROR_EMBEDDING_FAILED

    def test_without_detail(self):
        err = EmbeddingGenerationError()
        assert "Embedding generation failed" in err.message


class TestLLMGenerationError:
    def test_with_detail(self):
        err = LLMGenerationError("Timeout")
        assert "Timeout" in err.message
        assert err.error_code == ERROR_LLM_FAILED


class TestTTSGenerationError:
    def test_with_detail(self):
        err = TTSGenerationError("Voice not available")
        assert "Voice not available" in err.message
        assert err.error_code == ERROR_TTS_FAILED


class TestAudioNotFoundError:
    def test_message_includes_id(self):
        err = AudioNotFoundError("audio-789")
        assert "audio-789" in err.message
        assert err.status_code == 404


class TestInvalidQueryModeError:
    def test_message_includes_mode_and_valid(self):
        err = InvalidQueryModeError("chat", ["explain", "teach"])
        assert "chat" in err.message
        assert "explain" in err.message
        assert err.status_code == 400
        assert err.error_code == ERROR_INVALID_QUERY_MODE
