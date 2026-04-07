"""
TTS Service

Converts text to speech using a pluggable TTS engine architecture.

Supported engines (selected via TTS_ENGINE config / env var):
- edge-tts   : Microsoft Edge Neural TTS (free, cloud, DEFAULT)
- coqui      : Coqui Glow-TTS (local, lightweight)
- xtts_v2    : Coqui XTTS v2 (multilingual, voice cloning)
- fast_pitch : NVIDIA FastPitch (ultra-fast, low latency)
- tortoise   : Tortoise TTS (highest quality, voice cloning)
- bark       : Suno Bark (expressive, multilingual)
"""

import logging
import uuid
from pathlib import Path
from typing import Optional

from app.core.config import get_settings, AUDIO_DIR
from app.core.constants import (
    TTS_ENGINE_EDGE,
    TTS_ENGINE_COQUI,
    TTS_ENGINE_XTTS_V2,
    TTS_ENGINE_FAST_PITCH,
    TTS_ENGINE_TORTOISE,
    TTS_ENGINE_BARK,
    SUPPORTED_TTS_ENGINES,
)
from app.services.audio_generation.speech_style_manager import (
    get_speech_style_manager,
    SpeechStyle,
)
from app.services.audio_generation.tts_models.base_tts_model import BaseTTSModel
from app.utils.text_utils import prepare_text_for_tts
from app.utils.error_handlers import TTSGenerationError

logger = logging.getLogger(__name__)


# ── Handler Factory ──────────────────────────────────────────

def _create_handler(engine: str) -> BaseTTSModel:
    """
    Instantiate the correct TTS handler based on the engine name.

    This factory is intentionally kept inside the module so that
    heavyweight imports (torch, TTS, bark …) only happen for the
    engine that is actually selected.
    """
    if engine == TTS_ENGINE_EDGE:
        from app.services.audio_generation.tts_models.edge_tts_handler import EdgeTTSHandler
        return EdgeTTSHandler()

    if engine == TTS_ENGINE_COQUI:
        from app.services.audio_generation.tts_models.coqui_tts_handler import CoquiTTSHandler
        return CoquiTTSHandler()

    if engine == TTS_ENGINE_XTTS_V2:
        from app.services.audio_generation.tts_models.xtts_v2_handler import XTTSv2Handler
        return XTTSv2Handler()

    if engine == TTS_ENGINE_FAST_PITCH:
        from app.services.audio_generation.tts_models.fast_pitch_handler import FastPitchHandler
        return FastPitchHandler()

    if engine == TTS_ENGINE_TORTOISE:
        from app.services.audio_generation.tts_models.tortoise_handler import TortoiseTTSHandler
        return TortoiseTTSHandler()

    if engine == TTS_ENGINE_BARK:
        from app.services.audio_generation.tts_models.bark_handler import BarkHandler
        return BarkHandler()

    raise TTSGenerationError(
        f"Unknown TTS engine: '{engine}'. "
        f"Supported engines: {SUPPORTED_TTS_ENGINES}"
    )


class TTSService:
    """
    Text-to-speech synthesis service.

    Delegates actual audio generation to the handler selected via
    the ``TTS_ENGINE`` configuration setting.  The handler is created
    once and reused for all requests.
    """

    def __init__(self):
        self.settings = get_settings()
        self.engine = self.settings.tts_engine.lower()
        self.style_manager = get_speech_style_manager()
        self.output_dir = AUDIO_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create the engine handler (model will be loaded lazily)
        self._handler: BaseTTSModel = _create_handler(self.engine)

        logger.info(
            f"TTSService initialized | engine={self.engine} "
            f"cloning={self._handler.supports_voice_cloning} "
            f"languages={self._handler.supported_languages}"
        )

    # ── Public Properties ────────────────────────────────────

    @property
    def handler(self) -> BaseTTSModel:
        """Return the active TTS engine handler."""
        return self._handler

    # ── Main Synthesis ───────────────────────────────────────

    async def synthesize(
        self,
        text: str,
        mode: str = "explain",
        voice: Optional[str] = None,
        rate: Optional[str] = None,
        output_format: str = "mp3",
        speaker_wav: Optional[str] = None,
        language: str = "en",
    ) -> dict:
        """
        Convert text to speech audio.

        Args:
            text:          The text to convert.
            mode:          'explain' or 'teach' (affects voice style).
            voice:         Override default voice.
            rate:          Override speech rate.
            output_format: Audio format (mp3, wav).
            speaker_wav:   Path to reference speaker audio (voice cloning).
            language:      ISO-639-1 language code.

        Returns:
            Dict with audio_id, file_path, duration_seconds, metadata.
        """
        if not text or not text.strip():
            raise TTSGenerationError("Empty text provided for TTS")

        # Prepare text for audio (remove markdown, URLs, etc.)
        clean_text = prepare_text_for_tts(text)

        # Get speech style
        style = self.style_manager.get_custom_style(mode, voice, rate)

        # Generate unique filename
        audio_id = str(uuid.uuid4())
        filename = f"{audio_id}.{output_format}"
        output_path = self.output_dir / filename

        logger.info(
            f"Synthesizing speech | id={audio_id} engine={self.engine} "
            f"voice={style.voice} chars={len(clean_text)}"
        )

        try:
            await self._handler.synthesize(
                text=clean_text,
                style=style,
                output_path=str(output_path),
                speaker_wav=speaker_wav,
                language=language,
            )
        except TTSGenerationError:
            raise
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            raise TTSGenerationError(str(e))

        # Get file info
        file_size = output_path.stat().st_size if output_path.exists() else 0
        duration = self._estimate_duration(clean_text)

        result = {
            "audio_id": audio_id,
            "file_path": str(output_path),
            "filename": filename,
            "format": output_format,
            "duration_seconds": duration,
            "file_size_bytes": file_size,
            "voice": style.voice,
            "mode": mode,
            "engine": self.engine,
            "voice_cloned": speaker_wav is not None,
        }

        logger.info(
            f"TTS complete | id={audio_id} engine={self.engine} "
            f"size={file_size} duration≈{duration:.1f}s"
        )
        return result

    # ── Engine Info ──────────────────────────────────────────

    def get_engine_info(self) -> dict:
        """Return information about the active TTS engine."""
        from app.core.constants import TTS_ENGINE_REGISTRY

        registry = TTS_ENGINE_REGISTRY.get(self.engine, {})
        return {
            **self._handler.get_model_info(),
            "inference_speed": registry.get("inference_speed", "unknown"),
            "description": registry.get("description", ""),
        }

    # ── Helpers ──────────────────────────────────────────────

    @staticmethod
    def _estimate_duration(text: str) -> float:
        """
        Estimate audio duration in seconds.
        Average speech rate: ~150 words per minute.
        """
        word_count = len(text.split())
        return (word_count / 150) * 60

    def get_audio_file(self, audio_id: str, fmt: str = "mp3") -> Optional[Path]:
        """Retrieve an audio file path by ID."""
        path = self.output_dir / f"{audio_id}.{fmt}"
        if path.exists():
            return path
        # Try other formats
        for f in ["mp3", "wav", "ogg"]:
            alt = self.output_dir / f"{audio_id}.{f}"
            if alt.exists():
                return alt
        return None

    def delete_audio(self, audio_id: str) -> bool:
        """Delete an audio file."""
        for fmt in ["mp3", "wav", "ogg"]:
            path = self.output_dir / f"{audio_id}.{fmt}"
            if path.exists():
                path.unlink()
                logger.info(f"Audio deleted | id={audio_id}")
                return True
        return False

    def list_audio_files(self) -> list[dict]:
        """List all generated audio files."""
        files = []
        for path in self.output_dir.iterdir():
            if path.suffix in [".mp3", ".wav", ".ogg"]:
                files.append({
                    "audio_id": path.stem,
                    "filename": path.name,
                    "format": path.suffix.lstrip("."),
                    "file_size_bytes": path.stat().st_size,
                })
        return files


# ── Module-level factory ─────────────────────────────────────

_tts_service: Optional[TTSService] = None


def get_tts_service() -> TTSService:
    global _tts_service
    if _tts_service is None:
        _tts_service = TTSService()
    return _tts_service
