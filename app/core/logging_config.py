"""
Centralized Logging Configuration

Provides consistent structured logging across all services.
"""

import logging
import sys
from app.core.config import get_settings


def setup_logging() -> None:
    """Configure application-wide logging based on environment settings."""
    settings = get_settings()

    log_format = (
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    date_format = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Reduce noise from third-party libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info(
        f"Logging initialized | env={settings.app_environment} "
        f"level={settings.log_level}"
    )


def get_logger(name: str) -> logging.Logger:
    """Return a logger with the given name (convenience wrapper)."""
    return logging.getLogger(name)
