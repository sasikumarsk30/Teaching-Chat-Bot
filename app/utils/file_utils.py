"""
File Utilities

Helpers for file operations, type detection, and validation.
"""

import logging
from pathlib import Path
from typing import Optional

from app.core.config import get_settings
from app.core.constants import SUPPORTED_FILE_TYPES
from app.utils.error_handlers import UnsupportedFileTypeError, FileTooLargeError

logger = logging.getLogger(__name__)


def validate_file(filename: str, file_size: int) -> str:
    """
    Validate file type and size.

    Returns:
        The detected file extension (e.g. '.pdf')

    Raises:
        UnsupportedFileTypeError: If file type is not allowed.
        FileTooLargeError: If file exceeds max size.
    """
    settings = get_settings()
    ext = get_file_extension(filename)

    if ext not in settings.allowed_extensions:
        raise UnsupportedFileTypeError(ext, settings.allowed_extensions)

    if file_size > settings.max_upload_size_bytes:
        raise FileTooLargeError(file_size, settings.max_upload_size_bytes)

    return ext


def get_file_extension(filename: str) -> str:
    """Return lowercase file extension including the dot."""
    return Path(filename).suffix.lower()


def get_mime_type(filename: str) -> str:
    """Return MIME type based on file extension."""
    ext = get_file_extension(filename)
    return SUPPORTED_FILE_TYPES.get(ext, "application/octet-stream")


def sanitize_filename(filename: str) -> str:
    """
    Remove potentially dangerous characters from a filename.
    Preserves the extension.
    """
    stem = Path(filename).stem
    ext = Path(filename).suffix

    # Keep only alphanumeric, hyphens, underscores, dots, spaces
    safe = "".join(
        c if c.isalnum() or c in "-_. " else "_" for c in stem
    )
    return f"{safe}{ext}"


def ensure_directory(directory: Path) -> Path:
    """Create directory if it doesn't exist. Returns the path."""
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_file_size_human(size_bytes: int) -> str:
    """Convert bytes to human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"
