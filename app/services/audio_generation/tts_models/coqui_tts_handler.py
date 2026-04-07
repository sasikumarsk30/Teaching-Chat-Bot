"""
Coqui TTS Handler

Coqui TTS (Glow-TTS) – fully local, open-source TTS engine.
"""

import asyncio
import logging
from typing import Optional

from app.core.config import get_settings
from app.core.constants import TTS_ENGINE_COQUI, TTS_ENGINE_REGISTRY
from app.services.audio_generation.tts_models.base_tts_model import BaseTTSModel
from app.services.audio_generation.speech_style_manager import SpeechStyle
from app.utils.error_handlers import TTSGenerationError

logger = logging.getLogger(__name__)

_REGISTRY = TTS_ENGINE_REGISTRY[TTS_ENGINE_COQUI]


class CoquiTTSHandler(BaseTTSModel):
    """Coqui Glow-TTS – local open-source text-to-speech."""

    def __init__(self):
        self._model = None
        cfg = get_settings().get_active_tts_config()
        self._model_name = cfg["model_name"]
        logger.info(f"CoquiTTSHandler created | model={self._model_name}")

    # ── Identity (driven by registry) ────────────────────────

    @property
    def engine_name(self) -> str:
        return TTS_ENGINE_COQUI

    @property
    def supports_voice_cloning(self) -> bool:
        return _REGISTRY["supports_voice_cloning"]

    @property
    def supported_languages(self) -> list[str]:
        return _REGISTRY["supported_languages"]

    # ── Lifecycle ────────────────────────────────────────────

    def load_model(self) -> None:
        """Download and initialise the Coqui Glow-TTS model."""
        if self._model is not None:
            return
        try:
            from TTS.api import TTS as CoquiTTS

            self._model = CoquiTTS(model_name=self._model_name)
            logger.info(f"Coqui TTS model loaded: {self._model_name}")
        except ImportError:
            raise TTSGenerationError(
                f"Coqui TTS not installed. Run: {_REGISTRY['install_hint']}"
            )
        except Exception as e:
            logger.error(f"Failed to load Coqui model: {e}")
            raise TTSGenerationError(f"Coqui model load failed: {e}")

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
        """Generate audio using the local Coqui Glow-TTS model."""
        if not self.is_model_loaded():
            self.load_model()

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self._model.tts_to_file(text=text, file_path=output_path),
            )
            logger.info(f"Coqui TTS saved to {output_path}")

        except Exception as e:
            logger.error(f"Coqui TTS synthesis failed: {e}")
            raise TTSGenerationError(f"Coqui TTS failed: {e}")
