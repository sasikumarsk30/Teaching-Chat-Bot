"""
FastPitch Handler

NVIDIA FastPitch – ultra-fast TTS with pitch control.
Best for low-latency / real-time synthesis.  English only.
"""

import asyncio
import logging
from typing import Optional

from app.core.config import get_settings, MODELS_DIR
from app.core.constants import TTS_ENGINE_FAST_PITCH, TTS_ENGINE_REGISTRY
from app.services.audio_generation.tts_models.base_tts_model import BaseTTSModel
from app.services.audio_generation.speech_style_manager import SpeechStyle
from app.utils.error_handlers import TTSGenerationError

logger = logging.getLogger(__name__)

_REGISTRY = TTS_ENGINE_REGISTRY[TTS_ENGINE_FAST_PITCH]


class FastPitchHandler(BaseTTSModel):
    """NVIDIA FastPitch – real-time, pitch-controllable TTS."""

    def __init__(self):
        cfg = get_settings().get_active_tts_config()
        self._model = None
        self._model_name = cfg["model_name"]
        self._cache_dir = str(MODELS_DIR / "tts")
        logger.info(f"FastPitchHandler created | model={self._model_name}")

    # ── Identity (driven by registry) ────────────────────────

    @property
    def engine_name(self) -> str:
        return TTS_ENGINE_FAST_PITCH

    @property
    def supports_voice_cloning(self) -> bool:
        return _REGISTRY["supports_voice_cloning"]

    @property
    def supported_languages(self) -> list[str]:
        return _REGISTRY["supported_languages"]

    # ── Lifecycle ────────────────────────────────────────────

    def load_model(self) -> None:
        """Download and initialise the FastPitch model."""
        if self._model is not None:
            return
        try:
            from TTS.api import TTS as CoquiTTS

            self._model = CoquiTTS(model_name=self._model_name)
            logger.info(f"FastPitch model loaded: {self._model_name}")
        except ImportError:
            raise TTSGenerationError(
                f"Coqui TTS not installed. Run: {_REGISTRY['install_hint']}"
            )
        except Exception as e:
            logger.error(f"Failed to load FastPitch: {e}")
            raise TTSGenerationError(f"FastPitch model load failed: {e}")

    def is_model_loaded(self) -> bool:
        return self._model is not None

    # ── Synthesis ────────────────────────────────────────────

    async def synthesize(
        self,
        text: str,
        style: SpeechStyle,
        output_path: str,
        speaker_wav: Optional[str] = None,
        language: str = "en",
    ) -> None:
        """Generate audio using the FastPitch model."""
        if not self.is_model_loaded():
            self.load_model()

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self._model.tts_to_file(text=text, file_path=output_path),
            )
            logger.info(f"FastPitch TTS saved to {output_path}")

        except Exception as e:
            logger.error(f"FastPitch synthesis failed: {e}")
            raise TTSGenerationError(f"FastPitch TTS failed: {e}")
