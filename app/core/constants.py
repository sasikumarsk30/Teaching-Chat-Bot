"""
Global Constants

Centralized constants used across the application.
"""

# ── Document Processing ──────────────────────────────────────

SUPPORTED_FILE_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".txt": "text/plain",
    ".md": "text/markdown",
}

MAX_DOCUMENT_TITLE_LENGTH = 255
MAX_CHUNK_CONTENT_LENGTH = 50_000

# ── Chunking Strategies ─────────────────────────────────────

CHUNKING_STRATEGY_SEMANTIC = "semantic"
CHUNKING_STRATEGY_FIXED = "fixed"
CHUNKING_STRATEGY_PARAGRAPH = "paragraph"

DEFAULT_CHUNKING_STRATEGY = CHUNKING_STRATEGY_SEMANTIC

# Sentence boundary markers for semantic chunking
SENTENCE_DELIMITERS = [".", "!", "?", "。", "！", "？"]
PARAGRAPH_DELIMITERS = ["\n\n", "\r\n\r\n"]

# ── Embedding ────────────────────────────────────────────────

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_DIMENSION = 384

# ── Query Modes ──────────────────────────────────────────────

QUERY_MODE_EXPLAIN = "explain"
QUERY_MODE_TEACH = "teach"
VALID_QUERY_MODES = [QUERY_MODE_EXPLAIN, QUERY_MODE_TEACH]

# ── Audio ────────────────────────────────────────────────────

AUDIO_FORMAT_MP3 = "mp3"
AUDIO_FORMAT_WAV = "wav"
AUDIO_FORMAT_OGG = "ogg"
SUPPORTED_AUDIO_FORMATS = [AUDIO_FORMAT_MP3, AUDIO_FORMAT_WAV, AUDIO_FORMAT_OGG]

DEFAULT_AUDIO_FORMAT = AUDIO_FORMAT_MP3

# TTS Engine / Model Types
TTS_ENGINE_EDGE = "edge-tts"
TTS_ENGINE_COQUI = "coqui"
TTS_ENGINE_XTTS_V2 = "xtts_v2"
TTS_ENGINE_FAST_PITCH = "fast_pitch"
TTS_ENGINE_TORTOISE = "tortoise"
TTS_ENGINE_BARK = "bark"

SUPPORTED_TTS_ENGINES = [
    TTS_ENGINE_EDGE,
    TTS_ENGINE_COQUI,
    TTS_ENGINE_XTTS_V2,
    TTS_ENGINE_FAST_PITCH,
    TTS_ENGINE_TORTOISE,
    TTS_ENGINE_BARK,
]

# ── TTS Engine Registry ─────────────────────────────────────
# Centralised metadata for every supported TTS engine.
# Each handler reads its capabilities from here instead of
# hard-coding them, ensuring a single source of truth.

TTS_ENGINE_REGISTRY: dict[str, dict] = {
    TTS_ENGINE_EDGE: {
        "model_name": None,                        # cloud – no local model
        "supports_voice_cloning": False,
        "supported_languages": [
            "en", "es", "fr", "de", "it", "pt", "nl", "pl", "ru",
            "zh", "ja", "ko", "ar", "hi", "tr", "sv", "da", "fi",
        ],
        "inference_speed": "fast",
        "description": "Microsoft Edge Neural TTS (cloud, free)",
        "install_hint": "pip install edge-tts",
    },
    TTS_ENGINE_COQUI: {
        "model_name": "tts_models/en/ljspeech/glow-tts",
        "supports_voice_cloning": False,
        "supported_languages": ["en"],
        "inference_speed": "medium",
        "description": "Coqui Glow-TTS (local, lightweight)",
        "install_hint": "pip install TTS",
    },
    TTS_ENGINE_XTTS_V2: {
        "model_name": "tts_models/multilingual/multi-dataset/xtts_v2",
        "supports_voice_cloning": True,
        "supported_languages": [
            "en", "es", "fr", "de", "it", "pt", "pl", "tr",
            "ru", "nl", "cs", "ar", "zh",
        ],
        "inference_speed": "fast",
        "description": "Coqui XTTS v2 (multilingual, voice cloning)",
        "install_hint": "pip install TTS>=0.22.0",
    },
    TTS_ENGINE_FAST_PITCH: {
        "model_name": "tts_models/en/ljspeech/fast_pitch",
        "supports_voice_cloning": False,
        "supported_languages": ["en"],
        "inference_speed": "very_fast",
        "description": "NVIDIA FastPitch (ultra-fast, low latency)",
        "install_hint": "pip install TTS",
    },
    TTS_ENGINE_TORTOISE: {
        "model_name": "tts_models/en/tortoise/tortoise-v2",
        "supports_voice_cloning": True,
        "supported_languages": ["en"],
        "inference_speed": "slow",
        "description": "Tortoise TTS (highest quality, voice cloning)",
        "install_hint": "pip install TTS",
    },
    TTS_ENGINE_BARK: {
        "model_name": "suno/bark",
        "supports_voice_cloning": False,
        "supported_languages": [
            "en", "es", "fr", "de", "it", "pt", "pl", "tr",
            "ru", "nl", "zh", "ja", "ko", "hi",
        ],
        "inference_speed": "medium",
        "description": "Suno Bark (expressive, multilingual)",
        "install_hint": "pip install bark scipy",
    },
}

# TTS voice styles
TTS_STYLE_NEUTRAL = "neutral"
TTS_STYLE_TEACHING = "teaching"
TTS_STYLE_EXPLAINING = "explaining"

# ── DuckDB Table Names ──────────────────────────────────────

TABLE_DOCUMENTS = "documents"
TABLE_CHUNKS = "chunks"
TABLE_CHUNK_VECTORS = "chunk_vectors"
TABLE_RESPONSE_CACHE = "response_cache"

# ── Parquet File Names ───────────────────────────────────────

PARQUET_CHUNKS_METADATA = "chunks_metadata.parquet"
PARQUET_CHUNKS_VECTORS = "chunks_vectors.parquet"
PARQUET_DOCUMENTS_INDEX = "documents_index.parquet"

# ── API ──────────────────────────────────────────────────────

API_V1_PREFIX = "/api/v1"

# ── Pagination ───────────────────────────────────────────────

DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100

# ── Error Codes ──────────────────────────────────────────────

ERROR_DOCUMENT_NOT_FOUND = "DOCUMENT_NOT_FOUND"
ERROR_CHUNK_NOT_FOUND = "CHUNK_NOT_FOUND"
ERROR_UNSUPPORTED_FILE_TYPE = "UNSUPPORTED_FILE_TYPE"
ERROR_FILE_TOO_LARGE = "FILE_TOO_LARGE"
ERROR_EMBEDDING_FAILED = "EMBEDDING_GENERATION_FAILED"
ERROR_LLM_FAILED = "LLM_GENERATION_FAILED"
ERROR_TTS_FAILED = "TTS_GENERATION_FAILED"
ERROR_AUDIO_NOT_FOUND = "AUDIO_NOT_FOUND"
ERROR_INVALID_QUERY_MODE = "INVALID_QUERY_MODE"
