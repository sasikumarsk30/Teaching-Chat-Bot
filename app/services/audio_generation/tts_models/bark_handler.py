"""
Bark Handler

Suno Bark – high-quality neural TTS with emotional expression
and environmental sounds.  Supports multiple languages and
various speaker presets.
"""

import asyncio
import logging
from typing import Optional

from app.core.config import get_settings, MODELS_DIR, AUDIO_DIR
from app.core.constants import TTS_ENGINE_BARK, TTS_ENGINE_REGISTRY
from app.services.audio_generation.tts_models.base_tts_model import BaseTTSModel
from app.services.audio_generation.speech_style_manager import SpeechStyle
from app.utils.error_handlers import TTSGenerationError

logger = logging.getLogger(__name__)

_REGISTRY = TTS_ENGINE_REGISTRY[TTS_ENGINE_BARK]


class BarkHandler(BaseTTSModel):
    """Suno Bark – expressive neural TTS with multi-language support."""

    def __init__(self):
        cfg = get_settings().get_active_tts_config()
        self._loaded = False
        self._bark_model_name = cfg["model_name"]
        self._default_speaker = cfg["bark_speaker"]
        logger.info(
            f"BarkHandler created | model={self._bark_model_name} "
            f"speaker={self._default_speaker}"
        )

    # ── Identity (driven by registry) ────────────────────────

    @property
    def engine_name(self) -> str:
        return TTS_ENGINE_BARK

    @property
    def supports_voice_cloning(self) -> bool:
        return _REGISTRY["supports_voice_cloning"]

    @property
    def supported_languages(self) -> list[str]:
        return _REGISTRY["supported_languages"]

    # ── Lifecycle ────────────────────────────────────────────

    def load_model(self) -> None:
        """Download and preload Bark models."""
        if self._loaded:
            return
        try:
            from bark import preload_models  # noqa: F401

            preload_models()
            self._loaded = True
            logger.info("Bark models preloaded")
        except ImportError:
            raise TTSGenerationError(
                f"Bark not installed. Run: {_REGISTRY['install_hint']}"
            )
        except Exception as e:
            logger.error(f"Failed to preload Bark: {e}")
            raise TTSGenerationError(f"Bark model load failed: {e}")

    def is_model_loaded(self) -> bool:
        return self._loaded

    # ── Synthesis ────────────────────────────────────────────

    async def synthesize(
        self,
        text: str,
        style: SpeechStyle,
        output_path: str,
        speaker_wav: Optional[str] = None,
        language: str = "en",
    ) -> None:
        """Generate speech using Bark with optional speaker preset."""
        if not self._loaded:
            self.load_model()

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._synthesize_sync,
                text,
                output_path,
            )
            logger.info(
                f"Bark TTS saved to {output_path} | "
                f"speaker={self._default_speaker}"
            )
        except TTSGenerationError:
            raise
        except Exception as e:
            logger.error(f"Bark synthesis failed: {e}")
            raise TTSGenerationError(f"Bark TTS failed: {e}")

    def _synthesize_sync(self, text: str, output_path: str) -> None:
        """Synchronous core synthesis (runs inside thread executor)."""
        try:
            from bark import generate_audio, SAMPLE_RATE
            import numpy as np
            from scipy.io.wavfile import write as write_wav

            audio_array = generate_audio(
                text,
                history_prompt=self._default_speaker,
            )

            # Bark outputs a numpy float32 array → write as WAV
            audio_int16 = (audio_array * 32767).astype(np.int16)
            write_wav(output_path, SAMPLE_RATE, audio_int16)

        except ImportError as e:
            raise TTSGenerationError(
                f"Missing dependency for Bark: {e}. "
                "Run: pip install bark scipy"
            )
        except Exception as e:
            raise TTSGenerationError(f"Bark synthesis error: {e}")
