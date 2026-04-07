"""
Model Manager

Handles downloading and caching of open-source models
(embeddings, TTS) on first use.
"""

import logging
from pathlib import Path
from typing import Optional

from app.core.config import MODELS_DIR, get_settings

logger = logging.getLogger(__name__)

EMBEDDING_MODELS_DIR = MODELS_DIR / "embeddings"
TTS_MODELS_DIR = MODELS_DIR / "tts"


class ModelManager:
    """Manages model downloads, caching, and availability checks."""

    def __init__(self):
        self.settings = get_settings()
        EMBEDDING_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        TTS_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("ModelManager initialized")

    def ensure_embedding_model(self) -> str:
        """
        Ensure the embedding model is available locally.
        Downloads via sentence-transformers on first call.

        Returns:
            The model name/path to load.
        """
        model_name = self.settings.embedding_model_name
        cache_dir = str(EMBEDDING_MODELS_DIR)

        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {model_name}")
            # SentenceTransformer will download to cache_dir if not present
            _ = SentenceTransformer(model_name, cache_folder=cache_dir)
            logger.info(f"Embedding model ready: {model_name}")
            return model_name
        except Exception as e:
            logger.error(f"Failed to load embedding model {model_name}: {e}")
            raise

    def ensure_tts_engine(self) -> str:
        """
        Ensure the TTS engine is available.
        For edge-tts, no download needed (uses Microsoft Edge API).
        For Coqui TTS, downloads model weights.

        Returns:
            The TTS engine name.
        """
        engine = self.settings.tts_engine.lower()

        if engine == "edge-tts":
            logger.info("TTS engine: edge-tts (no local model required)")
            return engine
        elif engine == "coqui":
            try:
                from TTS.api import TTS as CoquiTTS

                logger.info("Loading Coqui TTS model...")
                _ = CoquiTTS(model_name="tts_models/en/ljspeech/glow-tts")
                logger.info("Coqui TTS model ready")
                return engine
            except ImportError:
                logger.warning(
                    "Coqui TTS not installed. Falling back to edge-tts."
                )
                return "edge-tts"
            except Exception as e:
                logger.error(f"Failed to load Coqui TTS: {e}")
                raise
        else:
            logger.warning(f"Unknown TTS engine '{engine}', defaulting to edge-tts")
            return "edge-tts"

    def get_model_info(self) -> dict:
        """Return info about configured models."""
        return {
            "embedding_model": self.settings.embedding_model_name,
            "embedding_dimension": self.settings.embedding_dimension,
            "tts_engine": self.settings.tts_engine,
            "tts_voice": self.settings.edge_tts_voice,
            "llm_provider": self.settings.llm_provider,
            "llm_model": self.settings.llm_model_name,
        }


# ── Module-level factory ─────────────────────────────────────

_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager
