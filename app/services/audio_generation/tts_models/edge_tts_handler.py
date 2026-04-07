"""
Edge TTS Handler

Microsoft Edge Neural TTS engine (free, high quality, requires internet).
This is the DEFAULT engine for the application.
"""

import logging
from typing import Optional

from app.core.constants import TTS_ENGINE_EDGE, TTS_ENGINE_REGISTRY
from app.services.audio_generation.tts_models.base_tts_model import BaseTTSModel
from app.services.audio_generation.speech_style_manager import SpeechStyle
from app.utils.error_handlers import TTSGenerationError

logger = logging.getLogger(__name__)

_REGISTRY = TTS_ENGINE_REGISTRY[TTS_ENGINE_EDGE]


class EdgeTTSHandler(BaseTTSModel):
    """Microsoft Edge Neural TTS - cloud-based, free, high quality."""

    def __init__(self):
        self._loaded = False
        logger.info("EdgeTTSHandler created")

    # ── Identity (driven by registry) ────────────────────────

    @property
    def engine_name(self) -> str:
        return TTS_ENGINE_EDGE

    @property
    def supports_voice_cloning(self) -> bool:
        return _REGISTRY["supports_voice_cloning"]

    @property
    def supported_languages(self) -> list[str]:
        return _REGISTRY["supported_languages"]

    # ── Lifecycle ────────────────────────────────────────────

    def load_model(self) -> None:
        """Edge TTS has no local model to load; just verify the package."""
        try:
            import edge_tts  # noqa: F401

            self._loaded = True
            logger.info("EdgeTTSHandler ready (package verified)")
        except ImportError:
            raise TTSGenerationError(
                f"edge-tts package not installed. Run: {_REGISTRY['install_hint']}"
            )

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
        """Generate audio using Microsoft Edge Neural TTS."""
        if not self._loaded:
            self.load_model()

        try:
            import edge_tts

            communicate = edge_tts.Communicate(
                text=text,
                voice=style.voice,
                rate=style.rate,
                pitch=style.pitch,
                volume=style.volume,
            )
            await communicate.save(output_path)
            logger.info(f"Edge TTS saved to {output_path}")

        except ImportError:
            raise TTSGenerationError(
                f"edge-tts package not installed. Run: {_REGISTRY['install_hint']}"
            )
        except Exception as e:
            logger.error(f"Edge TTS synthesis failed: {e}")
            raise TTSGenerationError(f"Edge TTS failed: {e}")
