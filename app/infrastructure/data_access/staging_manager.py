"""
Staging Manager

Provides batch staging for chunks before embedding generation.
Collects chunks in-memory and flushes to Parquet + DuckDB when threshold is met.
"""

import json
import logging
import threading
from datetime import datetime
from typing import Optional

from app.core.config import get_settings
from app.infrastructure.data_access.duckdb_manager import get_duckdb_manager
from app.infrastructure.data_access.parquet_manager import get_parquet_manager
from app.core.constants import TABLE_CHUNKS

logger = logging.getLogger(__name__)


class StagingManager:
    """Thread-safe staging area for batch chunk processing."""

    def __init__(self, flush_threshold: int = 100, flush_timeout_seconds: int = 300):
        self.settings = get_settings()
        self.db = get_duckdb_manager()
        self.parquet = get_parquet_manager()

        self._staging_buffer: list[dict] = []
        self._lock = threading.Lock()
        self._flush_threshold = flush_threshold
        self._flush_timeout = flush_timeout_seconds
        self._flush_timer: Optional[threading.Timer] = None

        logger.info(
            f"StagingManager initialized | threshold={flush_threshold} "
            f"timeout={flush_timeout_seconds}s"
        )

    def add_chunks(self, chunks: list[dict]) -> int:
        """
        Add chunks to the staging buffer.
        Triggers auto-flush if threshold is exceeded.

        Returns:
            Number of chunks currently in the buffer.
        """
        with self._lock:
            self._staging_buffer.extend(chunks)
            buffer_size = len(self._staging_buffer)

            logger.info(
                f"Added {len(chunks)} chunks to staging | buffer={buffer_size}"
            )

            if buffer_size >= self._flush_threshold:
                self._flush_internal()
            elif self._flush_timer is None:
                # Start a timeout-based flush for smaller batches
                self._flush_timer = threading.Timer(
                    self._flush_timeout, self.flush
                )
                self._flush_timer.daemon = True
                self._flush_timer.start()

            return buffer_size

    def flush(self) -> int:
        """Force-flush all staged chunks to Parquet and DuckDB."""
        with self._lock:
            return self._flush_internal()

    def _flush_internal(self) -> int:
        """Internal flush (must be called with lock held)."""
        if not self._staging_buffer:
            return 0

        count = len(self._staging_buffer)

        try:
            import pandas as pd

            df = pd.DataFrame(self._staging_buffer)

            # Write to Parquet
            self.parquet.write_chunks_metadata(df)

            # Write to DuckDB
            for chunk in self._staging_buffer:
                self.db.execute(
                    f"""
                    INSERT OR REPLACE INTO {TABLE_CHUNKS}
                        (id, document_id, sequence, content,
                         chunk_size, start_char, end_char, created_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        chunk["id"],
                        chunk["document_id"],
                        chunk["sequence"],
                        chunk["content"],
                        chunk.get("chunk_size", len(chunk["content"])),
                        chunk.get("start_char", 0),
                        chunk.get("end_char", 0),
                        chunk.get("created_at", datetime.utcnow()),
                        json.dumps(chunk.get("metadata", {})),
                    ],
                )

            self._staging_buffer.clear()

            # Cancel any pending timer
            if self._flush_timer is not None:
                self._flush_timer.cancel()
                self._flush_timer = None

            logger.info(f"Staging flushed | chunks={count}")

        except Exception as e:
            logger.error(f"Staging flush failed: {e}")
            raise

        return count

    def get_buffer_size(self) -> int:
        """Return current number of staged chunks."""
        with self._lock:
            return len(self._staging_buffer)

    def clear(self) -> None:
        """Discard all staged data without flushing."""
        with self._lock:
            self._staging_buffer.clear()
            if self._flush_timer is not None:
                self._flush_timer.cancel()
                self._flush_timer = None
            logger.info("Staging buffer cleared")

    def shutdown(self) -> None:
        """Flush remaining data and clean up timers."""
        self.flush()
        logger.info("StagingManager shut down")


# ── Module-level factory ─────────────────────────────────────

_staging_manager: Optional[StagingManager] = None


def get_staging_manager() -> StagingManager:
    global _staging_manager
    if _staging_manager is None:
        _staging_manager = StagingManager()
    return _staging_manager
