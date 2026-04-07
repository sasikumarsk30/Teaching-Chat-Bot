"""
Input Validators

Additional validation logic beyond Pydantic model constraints.
"""

from app.core.constants import VALID_QUERY_MODES, SUPPORTED_AUDIO_FORMATS
from app.utils.error_handlers import InvalidQueryModeError


def validate_query_mode(mode: str) -> str:
    """
    Validate and normalize a query mode string.

    Raises:
        InvalidQueryModeError if invalid.
    """
    mode = mode.lower().strip()
    if mode not in VALID_QUERY_MODES:
        raise InvalidQueryModeError(mode, VALID_QUERY_MODES)
    return mode


def validate_audio_format(fmt: str) -> str:
    """Validate audio output format."""
    fmt = fmt.lower().strip()
    if fmt not in SUPPORTED_AUDIO_FORMATS:
        raise ValueError(
            f"Unsupported audio format: '{fmt}'. "
            f"Supported: {SUPPORTED_AUDIO_FORMATS}"
        )
    return fmt
