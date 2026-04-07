"""
Custom Exception Classes and Error Handlers

Provides structured error responses for the API.
"""

from fastapi import Request
from fastapi.responses import JSONResponse
from app.core.constants import (
    ERROR_DOCUMENT_NOT_FOUND,
    ERROR_UNSUPPORTED_FILE_TYPE,
    ERROR_FILE_TOO_LARGE,
    ERROR_EMBEDDING_FAILED,
    ERROR_LLM_FAILED,
    ERROR_TTS_FAILED,
    ERROR_AUDIO_NOT_FOUND,
    ERROR_CHUNK_NOT_FOUND,
    ERROR_INVALID_QUERY_MODE,
)


# ── Custom Exceptions ────────────────────────────────────────

class AppError(Exception):
    """Base application exception."""

    def __init__(self, message: str, error_code: str = "INTERNAL_ERROR", status_code: int = 500):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        super().__init__(self.message)


class DocumentNotFoundError(AppError):
    def __init__(self, doc_id: str):
        super().__init__(
            message=f"Document not found: {doc_id}",
            error_code=ERROR_DOCUMENT_NOT_FOUND,
            status_code=404,
        )


class ChunkNotFoundError(AppError):
    def __init__(self, chunk_id: str):
        super().__init__(
            message=f"Chunk not found: {chunk_id}",
            error_code=ERROR_CHUNK_NOT_FOUND,
            status_code=404,
        )


class UnsupportedFileTypeError(AppError):
    def __init__(self, file_type: str, allowed: list[str]):
        super().__init__(
            message=f"Unsupported file type: '{file_type}'. Allowed: {allowed}",
            error_code=ERROR_UNSUPPORTED_FILE_TYPE,
            status_code=400,
        )


class FileTooLargeError(AppError):
    def __init__(self, size_bytes: int, max_bytes: int):
        super().__init__(
            message=(
                f"File size ({size_bytes / 1024 / 1024:.1f} MB) exceeds "
                f"maximum ({max_bytes / 1024 / 1024:.1f} MB)"
            ),
            error_code=ERROR_FILE_TOO_LARGE,
            status_code=413,
        )


class EmbeddingGenerationError(AppError):
    def __init__(self, detail: str = ""):
        super().__init__(
            message=f"Embedding generation failed. {detail}".strip(),
            error_code=ERROR_EMBEDDING_FAILED,
            status_code=500,
        )


class LLMGenerationError(AppError):
    def __init__(self, detail: str = ""):
        super().__init__(
            message=f"LLM response generation failed. {detail}".strip(),
            error_code=ERROR_LLM_FAILED,
            status_code=500,
        )


class TTSGenerationError(AppError):
    def __init__(self, detail: str = ""):
        super().__init__(
            message=f"Text-to-speech generation failed. {detail}".strip(),
            error_code=ERROR_TTS_FAILED,
            status_code=500,
        )


class AudioNotFoundError(AppError):
    def __init__(self, audio_id: str):
        super().__init__(
            message=f"Audio file not found: {audio_id}",
            error_code=ERROR_AUDIO_NOT_FOUND,
            status_code=404,
        )


class InvalidQueryModeError(AppError):
    def __init__(self, mode: str, valid_modes: list[str]):
        super().__init__(
            message=f"Invalid query mode: '{mode}'. Valid modes: {valid_modes}",
            error_code=ERROR_INVALID_QUERY_MODE,
            status_code=400,
        )


# ── FastAPI Exception Handler ────────────────────────────────

async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """Global handler for AppError exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error_code": exc.error_code,
            "message": exc.message,
        },
    )


async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Fallback handler for unhandled exceptions."""
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error_code": "INTERNAL_ERROR",
            "message": "An unexpected error occurred.",
        },
    )
