"""
Base TTS Model

Abstract base class defining the interface that every TTS engine handler
must implement.  The TTSService delegates synthesis to whichever concrete
handler is selected via configuration.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

from app.services.audio_generation.speech_style_manager import SpeechStyle

logger = logging.getLogger(__name__)


class BaseTTSModel(ABC):
    """Abstract interface for TTS engine handlers."""

    # ── Identity ─────────────────────────────────────────────

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Short identifier for this engine (matches TTS_ENGINE constant)."""
        ...

    @property
    @abstractmethod
    def supports_voice_cloning(self) -> bool:
        """Whether this engine can clone a voice from a reference audio."""
        ...

    @property
    @abstractmethod
    def supported_languages(self) -> list[str]:
        """ISO-639-1 language codes this engine supports."""
        ...

    # ── Lifecycle ────────────────────────────────────────────

    @abstractmethod
    def load_model(self) -> None:
        """
        Download / initialise the underlying model.

        Called lazily on first synthesis request.  Implementations should
        be idempotent (safe to call more than once).
        """
        ...

    @abstractmethod
    def is_model_loaded(self) -> bool:
        """Return True if the model is ready for inference."""
        ...

    # ── Synthesis ────────────────────────────────────────────

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        style: SpeechStyle,
        output_path: str,
        speaker_wav: Optional[str] = None,
        language: str = "en",
    ) -> None:
        """
        Generate speech audio and write it to *output_path*.

        Args:
            text:        Clean text to synthesize.
            style:       Voice style parameters (voice, rate, pitch …).
            output_path: Destination file path for the audio.
            speaker_wav: Optional path to a reference speaker WAV file
                         (used by engines that support voice cloning).
            language:    ISO-639-1 language code.

        Raises:
            TTSGenerationError on failure.
        """
        ...

    # ── Info ─────────────────────────────────────────────────

    def get_model_info(self) -> dict:
        """Return a summary dict for monitoring / debugging."""
        return {
            "engine": self.engine_name,
            "loaded": self.is_model_loaded(),
            "supports_cloning": self.supports_voice_cloning,
            "languages": self.supported_languages,
        }
