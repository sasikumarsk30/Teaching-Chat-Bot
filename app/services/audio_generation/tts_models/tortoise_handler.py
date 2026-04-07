"""
Tortoise TTS Handler

Tortoise TTS – highest-quality open-source TTS with voice cloning.
Slower inference but produces extremely natural speech.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

from app.core.config import get_settings, MODELS_DIR
from app.core.constants import TTS_ENGINE_TORTOISE, TTS_ENGINE_REGISTRY
from app.services.audio_generation.tts_models.base_tts_model import BaseTTSModel
from app.services.audio_generation.speech_style_manager import SpeechStyle
from app.utils.error_handlers import TTSGenerationError

logger = logging.getLogger(__name__)

_REGISTRY = TTS_ENGINE_REGISTRY[TTS_ENGINE_TORTOISE]


class TortoiseTTSHandler(BaseTTSModel):
    """Tortoise TTS – ultra-high-quality speech synthesis with cloning."""

    # Quality presets (trade speed for quality)
    _VALID_PRESETS = ["ultra_fast", "fast", "standard", "high_quality"]

    def __init__(self):
        cfg = get_settings().get_active_tts_config()
        self._model = None
        self._model_name = cfg["model_name"]
        self._default_speaker_wav = cfg["speaker_wav"]
        self._preset = cfg["quality_preset"]
        self._cache_dir = str(MODELS_DIR / "tts")

        if self._preset not in self._VALID_PRESETS:
            logger.warning(
                f"Invalid Tortoise preset '{self._preset}', "
                f"falling back to 'fast'"
            )
            self._preset = "fast"

        logger.info(
            f"TortoiseTTSHandler created | model={self._model_name} "
            f"preset={self._preset}"
        )

    # ── Identity (driven by registry) ────────────────────────

    @property
    def engine_name(self) -> str:
        return TTS_ENGINE_TORTOISE

    @property
    def supports_voice_cloning(self) -> bool:
        return _REGISTRY["supports_voice_cloning"]

    @property
    def supported_languages(self) -> list[str]:
        return _REGISTRY["supported_languages"]

    # ── Lifecycle ────────────────────────────────────────────

    def load_model(self) -> None:
        """Download and initialise the Tortoise TTS model."""
        if self._model is not None:
            return
        try:
            from TTS.api import TTS as CoquiTTS

            self._model = CoquiTTS(model_name=self._model_name)
            logger.info(f"Tortoise model loaded: {self._model_name}")
        except ImportError:
            raise TTSGenerationError(
                f"Coqui TTS not installed. Run: {_REGISTRY['install_hint']}"
            )
        except Exception as e:
            logger.error(f"Failed to load Tortoise: {e}")
            raise TTSGenerationError(f"Tortoise model load failed: {e}")

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
        """
        Generate speech using Tortoise TTS.

        If *speaker_wav* is provided (or configured), the output audio
        will mimic the reference speaker's voice (voice cloning).
        """
        if not self.is_model_loaded():
            self.load_model()

        ref_wav = speaker_wav or self._default_speaker_wav

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._synthesize_sync,
                text,
                output_path,
                ref_wav,
            )
            logger.info(
                f"Tortoise TTS saved to {output_path} | "
                f"cloned={ref_wav is not None} preset={self._preset}"
            )
        except TTSGenerationError:
            raise
        except Exception as e:
            logger.error(f"Tortoise synthesis failed: {e}")
            raise TTSGenerationError(f"Tortoise TTS failed: {e}")

    def _synthesize_sync(
        self,
        text: str,
        output_path: str,
        speaker_wav: Optional[str],
    ) -> None:
        """Synchronous core synthesis (runs inside thread executor)."""
        kwargs: dict = {
            "text": text,
            "file_path": output_path,
        }

        if speaker_wav:
            wav_path = Path(speaker_wav)
            if not wav_path.exists():
                raise TTSGenerationError(
                    f"Speaker reference WAV not found: {speaker_wav}"
                )
            kwargs["speaker_wav"] = str(wav_path)

        self._model.tts_to_file(**kwargs)
