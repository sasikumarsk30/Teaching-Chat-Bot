"""
XTTS v2 Handler

Coqui XTTS v2 – multilingual, high-quality TTS with voice-cloning support.
Generates speech that can mimic a reference speaker from a short audio sample.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

from app.core.config import get_settings, MODELS_DIR
from app.core.constants import TTS_ENGINE_XTTS_V2, TTS_ENGINE_REGISTRY
from app.services.audio_generation.tts_models.base_tts_model import BaseTTSModel
from app.services.audio_generation.speech_style_manager import SpeechStyle
from app.utils.error_handlers import TTSGenerationError

logger = logging.getLogger(__name__)

_REGISTRY = TTS_ENGINE_REGISTRY[TTS_ENGINE_XTTS_V2]


class XTTSv2Handler(BaseTTSModel):
    """Coqui XTTS v2 – multilingual TTS with zero-shot voice cloning."""

    def __init__(self):
        cfg = get_settings().get_active_tts_config()
        self._model = None
        self._model_name = cfg["model_name"]
        self._use_gpu = cfg["use_gpu"]
        self._default_speaker_wav = cfg["speaker_wav"]
        self._default_language = cfg["language"]
        self._cache_dir = str(MODELS_DIR / "tts")
        logger.info(
            f"XTTSv2Handler created | model={self._model_name} "
            f"gpu={self._use_gpu} lang={self._default_language}"
        )

    # ── Identity (driven by registry) ────────────────────────

    @property
    def engine_name(self) -> str:
        return TTS_ENGINE_XTTS_V2

    @property
    def supports_voice_cloning(self) -> bool:
        return _REGISTRY["supports_voice_cloning"]

    @property
    def supported_languages(self) -> list[str]:
        return _REGISTRY["supported_languages"]

    # ── Lifecycle ────────────────────────────────────────────

    def load_model(self) -> None:
        """Download and initialise the XTTS v2 model."""
        if self._model is not None:
            return
        try:
            from TTS.api import TTS as CoquiTTS

            self._model = CoquiTTS(
                model_name=self._model_name,
                gpu=self._use_gpu,
            )
            logger.info(f"XTTS v2 model loaded: {self._model_name}")
        except ImportError:
            raise TTSGenerationError(
                f"Coqui TTS (>=0.22.0) not installed. Run: {_REGISTRY['install_hint']}"
            )
        except Exception as e:
            logger.error(f"Failed to load XTTS v2: {e}")
            raise TTSGenerationError(f"XTTS v2 model load failed: {e}")

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
        Generate speech using XTTS v2.

        If *speaker_wav* is provided (or configured), the output audio
        will mimic the reference speaker's voice (voice cloning).
        """
        if not self.is_model_loaded():
            self.load_model()

        ref_wav = speaker_wav or self._default_speaker_wav
        lang = language or self._default_language

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._synthesize_sync,
                text,
                output_path,
                ref_wav,
                lang,
            )
            logger.info(
                f"XTTS v2 saved to {output_path} | "
                f"cloned={ref_wav is not None} lang={lang}"
            )
        except TTSGenerationError:
            raise
        except Exception as e:
            logger.error(f"XTTS v2 synthesis failed: {e}")
            raise TTSGenerationError(f"XTTS v2 failed: {e}")

    def _synthesize_sync(
        self,
        text: str,
        output_path: str,
        speaker_wav: Optional[str],
        language: str,
    ) -> None:
        """Synchronous core synthesis (runs inside thread executor)."""
        kwargs: dict = {
            "text": text,
            "file_path": output_path,
            "language": language,
        }

        if speaker_wav:
            wav_path = Path(speaker_wav)
            if not wav_path.exists():
                raise TTSGenerationError(
                    f"Speaker reference WAV not found: {speaker_wav}"
                )
            kwargs["speaker_wav"] = str(wav_path)

        self._model.tts_to_file(**kwargs)
