"""
Application Configuration Module

Provides environment-specific configuration (DEV / QA / PROD)
loaded from environment variables with sensible defaults.
"""

import os
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


# ── Base Paths ────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

DOCUMENTS_DIR = DATA_DIR / "documents"
CHUNKS_DIR = DATA_DIR / "chunks"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
AUDIO_DIR = DATA_DIR / "audio"
DUCKDB_DIR = DATA_DIR / "duckdb"

# Ensure directories exist at import time
for _dir in [DOCUMENTS_DIR, CHUNKS_DIR, EMBEDDINGS_DIR, AUDIO_DIR, DUCKDB_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)


# ── Settings Class ────────────────────────────────────────────

class AppSettings(BaseSettings):
    """Central application settings backed by env vars / .env file."""

    # ── Environment ──────────────────────────────────────────
    app_environment: str = Field(default="DEV", alias="APP_ENVIRONMENT")
    app_name: str = "Document Audio Generation Service"
    app_version: str = "1.0.0"
    debug: bool = Field(default=True, alias="DEBUG")
    log_level: str = Field(default="DEBUG", alias="LOG_LEVEL")

    # ── Server ───────────────────────────────────────────────
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8001, alias="PORT")

    # ── Document Processing ──────────────────────────────────
    max_upload_size_mb: int = Field(default=50, alias="MAX_UPLOAD_SIZE_MB")
    allowed_file_types: str = Field(
        default=".pdf,.docx,.txt,.md",
        alias="ALLOWED_FILE_TYPES",
    )

    # ── Chunking ─────────────────────────────────────────────
    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")
    min_chunk_size: int = Field(default=100, alias="MIN_CHUNK_SIZE")

    # ── Embeddings ───────────────────────────────────────────
    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL_NAME",
    )
    embedding_dimension: int = Field(default=384, alias="EMBEDDING_DIMENSION")
    embedding_batch_size: int = Field(default=32, alias="EMBEDDING_BATCH_SIZE")

    # ── LLM ──────────────────────────────────────────────────
    llm_provider: str = Field(default="ollama", alias="LLM_PROVIDER")
    llm_model_name: str = Field(default="mistral", alias="LLM_MODEL_NAME")
    llm_base_url: str = Field(
        default="http://localhost:11434", alias="LLM_BASE_URL"
    )
    llm_temperature: float = Field(default=0.7, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=2048, alias="LLM_MAX_TOKENS")

    # ── TTS (Text-to-Speech) ─────────────────────────────────
    tts_engine: str = Field(default="edge-tts", alias="TTS_ENGINE")
    tts_output_format: str = Field(default="mp3", alias="TTS_OUTPUT_FORMAT")

    # Edge-TTS specific (cloud, default engine)
    edge_tts_voice: str = Field(
        default="en-US-AriaNeural", alias="EDGE_TTS_VOICE"
    )
    edge_tts_rate: str = Field(default="+0%", alias="EDGE_TTS_RATE")
    edge_tts_pitch: str = Field(default="+0Hz", alias="EDGE_TTS_PITCH")

    # ── Per-Engine Overrides ─────────────────────────────────
    # Each engine reads its model_name from TTS_ENGINE_REGISTRY by
    # default.  The env vars below let you override per-engine
    # settings without touching the registry.
    tts_model_name_override: Optional[str] = Field(
        default=None, alias="TTS_MODEL_NAME",
        description="Override the default model name for the active engine.",
    )
    tts_language: str = Field(default="en", alias="TTS_LANGUAGE")
    tts_speaker_wav: Optional[str] = Field(
        default=None, alias="TTS_SPEAKER_WAV",
        description="Path to reference speaker audio for voice cloning (XTTS v2 / Tortoise).",
    )
    tts_use_gpu: bool = Field(default=False, alias="TTS_USE_GPU")
    tts_quality_preset: str = Field(
        default="fast", alias="TTS_QUALITY_PRESET",
        description="Quality preset for Tortoise TTS: ultra_fast, fast, standard, high_quality.",
    )
    tts_bark_speaker: str = Field(
        default="v2/en_speaker_6", alias="TTS_BARK_SPEAKER",
        description="Bark speaker preset (e.g. v2/en_speaker_0 through v2/en_speaker_9).",
    )

    # ── DuckDB ───────────────────────────────────────────────
    duckdb_path: str = Field(
        default=str(DUCKDB_DIR / "document_index.duckdb"),
        alias="DUCKDB_PATH",
    )

    # ── Search / Retrieval ───────────────────────────────────
    search_top_k: int = Field(default=5, alias="SEARCH_TOP_K")
    similarity_threshold: float = Field(
        default=0.3, alias="SIMILARITY_THRESHOLD"
    )

    # ── Cache ────────────────────────────────────────────────
    cache_ttl_seconds: int = Field(default=3600, alias="CACHE_TTL_SECONDS")
    enable_response_cache: bool = Field(
        default=True, alias="ENABLE_RESPONSE_CACHE"
    )

    # ── Optional: Azure OpenAI (if using Azure instead of local) ─
    azure_openai_endpoint: Optional[str] = Field(
        default=None, alias="AZURE_OPENAI_ENDPOINT"
    )
    azure_openai_api_key: Optional[str] = Field(
        default=None, alias="AZURE_OPENAI_API_KEY"
    )
    azure_openai_api_version: str = Field(
        default="2024-02-15-preview", alias="AZURE_OPENAI_API_VERSION"
    )
    azure_openai_deployment: Optional[str] = Field(
        default=None, alias="AZURE_OPENAI_DEPLOYMENT"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        populate_by_name = True

    # ── Derived helpers ──────────────────────────────────────

    @property
    def allowed_extensions(self) -> list[str]:
        """Return allowed file extensions as a list."""
        return [ext.strip() for ext in self.allowed_file_types.split(",")]

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def is_dev(self) -> bool:
        return self.app_environment.upper() == "DEV"

    @property
    def is_prod(self) -> bool:
        return self.app_environment.upper() == "PROD"

    @property
    def uses_azure_openai(self) -> bool:
        return (
            self.azure_openai_endpoint is not None
            and self.azure_openai_api_key is not None
        )

    # ── TTS helpers ──────────────────────────────────────────

    def get_active_tts_config(self) -> dict:
        """
        Return the full configuration dict for the active TTS engine.

        Merges immutable metadata from ``TTS_ENGINE_REGISTRY`` with the
        user-overridable settings from env vars / .env.  If the user set
        ``TTS_MODEL_NAME`` it takes precedence over the registry default.

        Returns:
            Dict with keys: engine, model_name, supports_voice_cloning,
            supported_languages, inference_speed, description,
            install_hint, language, speaker_wav, use_gpu, quality_preset,
            bark_speaker, edge_tts_voice, edge_tts_rate, edge_tts_pitch.
        """
        from app.core.constants import TTS_ENGINE_REGISTRY

        engine = self.tts_engine.lower()
        registry = TTS_ENGINE_REGISTRY.get(engine, {})

        return {
            # ── Identity (from registry) ──────────────────────
            "engine": engine,
            "model_name": (
                self.tts_model_name_override
                or registry.get("model_name")
            ),
            "supports_voice_cloning": registry.get(
                "supports_voice_cloning", False
            ),
            "supported_languages": registry.get(
                "supported_languages", ["en"]
            ),
            "inference_speed": registry.get("inference_speed", "medium"),
            "description": registry.get("description", ""),
            "install_hint": registry.get("install_hint", ""),
            # ── User overrides (from env / settings) ─────────
            "language": self.tts_language,
            "speaker_wav": self.tts_speaker_wav,
            "use_gpu": self.tts_use_gpu,
            "quality_preset": self.tts_quality_preset,
            "bark_speaker": self.tts_bark_speaker,
            "output_format": self.tts_output_format,
            # ── Edge-TTS specific ────────────────────────────
            "edge_tts_voice": self.edge_tts_voice,
            "edge_tts_rate": self.edge_tts_rate,
            "edge_tts_pitch": self.edge_tts_pitch,
        }


# ── Environment-specific overrides ───────────────────────────

_ENV_OVERRIDES: dict[str, dict] = {
    "DEV": {
        "chunk_size": 1000,
        "embedding_batch_size": 16,
        "search_top_k": 5,
        "cache_ttl_seconds": 3600,
        "log_level": "DEBUG",
        "tts_engine": "coqui",           # fast, no local model
    },
    "QA": {
        "chunk_size": 2000,
        "embedding_batch_size": 32,
        "search_top_k": 8,
        "cache_ttl_seconds": 7200,
        "log_level": "INFO",
        "tts_engine": "edge-tts",           # balanced quality + speed
    },
    "PROD": {
        "chunk_size": 3000,
        "embedding_batch_size": 64,
        "search_top_k": 10,
        "cache_ttl_seconds": 86400,
        "log_level": "WARNING",
        "tts_engine": "edge-tts",           # recommended default
    },
}


@lru_cache()
def get_settings() -> AppSettings:
    """
    Return a cached singleton of AppSettings.

    Environment-specific overrides are applied on top of the base settings
    only when the corresponding env var is NOT already set.
    """
    settings = AppSettings()
    env = settings.app_environment.upper()
    overrides = _ENV_OVERRIDES.get(env, {})

    for key, value in overrides.items():
        # Only override if env var was not explicitly set
        env_var = key.upper()
        if os.getenv(env_var) is None:
            setattr(settings, key, value)

    return settings
