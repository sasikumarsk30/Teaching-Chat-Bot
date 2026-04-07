"""
Response Cache

Caches generated LLM responses and audio metadata
to avoid re-generating the same content.
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional

from app.core.config import get_settings
from app.core.constants import TABLE_RESPONSE_CACHE
from app.infrastructure.data_access.duckdb_manager import get_duckdb_manager

logger = logging.getLogger(__name__)


class ResponseCache:
    """Caches query responses and associated audio paths in DuckDB."""

    def __init__(self):
        self.settings = get_settings()
        self.db = get_duckdb_manager()
        self.ttl = self.settings.cache_ttl_seconds
        self.enabled = self.settings.enable_response_cache
        logger.info(
            f"ResponseCache initialized | enabled={self.enabled} ttl={self.ttl}s"
        )

    def get(self, query: str, mode: str) -> Optional[dict]:
        """Look up a cached response for a query + mode combo."""
        if not self.enabled:
            return None

        result = self.db.fetch_one(
            f"""
            SELECT * FROM {TABLE_RESPONSE_CACHE}
            WHERE query = ? AND mode = ? AND expires_at > ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            [query, mode, datetime.utcnow()],
        )

        if result:
            logger.info(f"Response cache HIT | query='{query[:50]}...' mode={mode}")
        return result

    def put(
        self,
        query: str,
        mode: str,
        response_text: str,
        source_chunk_ids: list[str] | None = None,
        audio_path: str | None = None,
        audio_duration_seconds: float | None = None,
    ) -> str:
        """
        Store a generated response in the cache.

        Returns:
            The generated cache entry ID (also usable as response_id).
        """
        if not self.enabled:
            return str(uuid.uuid4())

        entry_id = str(uuid.uuid4())
        expires = datetime.utcnow() + timedelta(seconds=self.ttl)

        self.db.execute(
            f"""
            INSERT INTO {TABLE_RESPONSE_CACHE}
                (id, query, mode, response_text, source_chunk_ids,
                 audio_path, audio_duration_seconds, created_at, expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                entry_id,
                query,
                mode,
                response_text,
                source_chunk_ids or [],
                audio_path,
                audio_duration_seconds,
                datetime.utcnow(),
                expires,
            ],
        )

        logger.info(
            f"Response cached | id={entry_id} query='{query[:50]}...' mode={mode}"
        )
        return entry_id

    def get_by_id(self, response_id: str) -> Optional[dict]:
        """Retrieve a cached response by its ID."""
        return self.db.fetch_one(
            f"SELECT * FROM {TABLE_RESPONSE_CACHE} WHERE id = ?",
            [response_id],
        )

    def update_audio(
        self,
        response_id: str,
        audio_path: str,
        audio_duration_seconds: float | None = None,
    ) -> None:
        """Attach audio info to an existing cached response."""
        self.db.execute(
            f"""
            UPDATE {TABLE_RESPONSE_CACHE}
            SET audio_path = ?, audio_duration_seconds = ?
            WHERE id = ?
            """,
            [audio_path, audio_duration_seconds, response_id],
        )
        logger.info(f"Audio attached to response cache | id={response_id}")

    def cleanup_expired(self) -> int:
        """Remove expired entries. Returns count of deleted rows."""
        result = self.db.fetch_one(
            f"""
            SELECT COUNT(*) as cnt FROM {TABLE_RESPONSE_CACHE}
            WHERE expires_at <= ?
            """,
            [datetime.utcnow()],
        )
        count = result["cnt"] if result else 0

        if count > 0:
            self.db.execute(
                f"DELETE FROM {TABLE_RESPONSE_CACHE} WHERE expires_at <= ?",
                [datetime.utcnow()],
            )
            logger.info(f"Cleaned up {count} expired cache entries")

        return count


# ── Module-level factory ─────────────────────────────────────

_response_cache: Optional[ResponseCache] = None


def get_response_cache() -> ResponseCache:
    global _response_cache
    if _response_cache is None:
        _response_cache = ResponseCache()
    return _response_cache
