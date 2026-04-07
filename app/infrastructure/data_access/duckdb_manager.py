"""
DuckDB Manager

Manages the DuckDB connection, schema initialization,
and provides query execution helpers.
"""

import duckdb
import logging
import threading
from pathlib import Path
from typing import Any, Optional

from app.core.config import get_settings
from app.core.constants import (
    TABLE_DOCUMENTS,
    TABLE_CHUNKS,
    TABLE_CHUNK_VECTORS,
    TABLE_RESPONSE_CACHE,
)

logger = logging.getLogger(__name__)


class DuckDBManager:
    """Thread-safe singleton for DuckDB operations."""

    _instance: Optional["DuckDBManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "DuckDBManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._db_lock = threading.Lock()
        self.settings = get_settings()
        self.db_path = self.settings.duckdb_path
        self._connection: Optional[duckdb.DuckDBPyConnection] = None
        logger.info(f"DuckDBManager created | path={self.db_path}")

    # ── Connection Management ────────────────────────────────

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        """Lazy-initialize and return the DuckDB connection."""
        if self._connection is None:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            self._connection = duckdb.connect(self.db_path)
            logger.info(f"DuckDB connected | path={self.db_path}")
        return self._connection

    def initialize_schema(self) -> None:
        """Create all required tables if they don't exist."""
        with self._db_lock:
            conn = self.connection

            # Documents table
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {TABLE_DOCUMENTS} (
                    id VARCHAR PRIMARY KEY,
                    filename VARCHAR NOT NULL,
                    title VARCHAR NOT NULL,
                    description VARCHAR,
                    file_type VARCHAR NOT NULL,
                    file_size_bytes INTEGER,
                    original_path VARCHAR,
                    total_chunks INTEGER DEFAULT 0,
                    embeddings_generated BOOLEAN DEFAULT FALSE,
                    tags VARCHAR[],
                    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSON
                )
            """)

            # Chunks table
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {TABLE_CHUNKS} (
                    id VARCHAR PRIMARY KEY,
                    document_id VARCHAR NOT NULL,
                    sequence INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    chunk_size INTEGER,
                    start_char INTEGER,
                    end_char INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSON,
                    FOREIGN KEY (document_id) REFERENCES {TABLE_DOCUMENTS}(id)
                )
            """)

            # Chunk vectors table
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {TABLE_CHUNK_VECTORS} (
                    id VARCHAR PRIMARY KEY,
                    chunk_id VARCHAR NOT NULL,
                    embedding FLOAT[],
                    vector_dim INTEGER,
                    model_name VARCHAR,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (chunk_id) REFERENCES {TABLE_CHUNKS}(id)
                )
            """)

            # Response cache table
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {TABLE_RESPONSE_CACHE} (
                    id VARCHAR PRIMARY KEY,
                    query TEXT NOT NULL,
                    mode VARCHAR NOT NULL,
                    response_text TEXT NOT NULL,
                    source_chunk_ids VARCHAR[],
                    audio_path VARCHAR,
                    audio_duration_seconds FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP
                )
            """)

            logger.info("DuckDB schema initialized successfully")

    # ── Query Helpers ────────────────────────────────────────

    def execute(self, sql: str, params: list | None = None) -> Any:
        """Execute a SQL statement (thread-safe)."""
        with self._db_lock:
            try:
                if params:
                    return self.connection.execute(sql, params)
                return self.connection.execute(sql)
            except Exception as e:
                # Reset the connection to clear any dirty/pending query state
                logger.warning(f"DuckDB execute error, resetting connection: {e}")
                try:
                    self._connection.close()
                except Exception:
                    pass
                self._connection = None
                raise

    def fetch_all(self, sql: str, params: list | None = None) -> list[dict]:
        """Execute and return all rows as list of dicts."""
        with self._db_lock:
            if params:
                result = self.connection.execute(sql, params)
            else:
                result = self.connection.execute(sql)

            columns = [desc[0] for desc in result.description]
            rows = result.fetchall()
            return [dict(zip(columns, row)) for row in rows]

    def fetch_one(self, sql: str, params: list | None = None) -> Optional[dict]:
        """Execute and return a single row as dict, or None."""
        with self._db_lock:
            if params:
                result = self.connection.execute(sql, params)
            else:
                result = self.connection.execute(sql)

            columns = [desc[0] for desc in result.description]
            row = result.fetchone()
            if row is None:
                return None
            return dict(zip(columns, row))

    def fetch_df(self, sql: str, params: list | None = None):
        """Execute and return a pandas DataFrame."""
        with self._db_lock:
            if params:
                return self.connection.execute(sql, params).fetchdf()
            return self.connection.execute(sql).fetchdf()

    def table_exists(self, table_name: str) -> bool:
        """Check whether a table exists in the database."""
        result = self.fetch_one(
            "SELECT COUNT(*) as cnt FROM information_schema.tables "
            "WHERE table_name = ?",
            [table_name],
        )
        return result is not None and result["cnt"] > 0

    def row_count(self, table_name: str) -> int:
        """Return number of rows in a table."""
        result = self.fetch_one(f"SELECT COUNT(*) as cnt FROM {table_name}")
        return result["cnt"] if result else 0

    # ── Cleanup ──────────────────────────────────────────────

    def close(self) -> None:
        """Close the DuckDB connection."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None
            logger.info("DuckDB connection closed")

    def vacuum(self) -> None:
        """Optimize storage by running VACUUM."""
        with self._db_lock:
            self.connection.execute("VACUUM")
            logger.info("DuckDB vacuum completed")


# ── Module-level singleton ───────────────────────────────────

def get_duckdb_manager() -> DuckDBManager:
    """Return the DuckDBManager singleton."""
    return DuckDBManager()
