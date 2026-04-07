"""
Audio Processor

Post-processing for generated audio files:
- Format conversion
- Normalization
- Metadata embedding
"""

import logging
from pathlib import Path
from typing import Optional

from app.core.config import AUDIO_DIR

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Post-processes audio files after TTS generation."""

    def __init__(self):
        self.output_dir = AUDIO_DIR
        logger.info("AudioProcessor initialized")

    def get_audio_metadata(self, file_path: str) -> dict:
        """
        Extract metadata from an audio file.

        Returns:
            Dict with duration, sample_rate, channels, file_size.
        """
        path = Path(file_path)
        if not path.exists():
            return {}

        metadata = {
            "file_path": str(path),
            "filename": path.name,
            "format": path.suffix.lstrip("."),
            "file_size_bytes": path.stat().st_size,
        }

        # Try to get duration using mutagen (lightweight metadata reader)
        try:
            duration = self._get_duration(file_path)
            metadata["duration_seconds"] = duration
        except Exception:
            pass

        return metadata

    def _get_duration(self, file_path: str) -> Optional[float]:
        """Get audio duration in seconds."""
        try:
            from mutagen.mp3 import MP3

            audio = MP3(file_path)
            return audio.info.length
        except ImportError:
            # Fallback: estimate from file size (rough: 128kbps MP3)
            size = Path(file_path).stat().st_size
            return size / (128 * 1024 / 8)  # bytes / (kbps * 1024 / 8)
        except Exception:
            return None

    def convert_format(
        self, input_path: str, target_format: str
    ) -> Optional[str]:
        """
        Convert audio to a different format using pydub.

        Args:
            input_path: Path to the source audio file.
            target_format: Target format (mp3, wav, ogg).

        Returns:
            Path to the converted file, or None on failure.
        """
        try:
            from pydub import AudioSegment

            audio = AudioSegment.from_file(input_path)
            output_path = Path(input_path).with_suffix(f".{target_format}")
            audio.export(str(output_path), format=target_format)
            logger.info(f"Audio converted: {input_path} → {output_path}")
            return str(output_path)

        except ImportError:
            logger.warning("pydub not installed. Format conversion unavailable.")
            return None
        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            return None

    def normalize_audio(self, file_path: str) -> Optional[str]:
        """
        Normalize audio volume levels.

        Returns:
            Path to the normalized file, or None on failure.
        """
        try:
            from pydub import AudioSegment
            from pydub.effects import normalize

            audio = AudioSegment.from_file(file_path)
            normalized = normalize(audio)
            normalized.export(file_path, format=Path(file_path).suffix.lstrip("."))
            logger.info(f"Audio normalized: {file_path}")
            return file_path

        except ImportError:
            logger.warning("pydub not installed. Normalization unavailable.")
            return None
        except Exception as e:
            logger.error(f"Audio normalization failed: {e}")
            return None


# ── Module-level factory ─────────────────────────────────────

_audio_processor: Optional[AudioProcessor] = None


def get_audio_processor() -> AudioProcessor:
    global _audio_processor
    if _audio_processor is None:
        _audio_processor = AudioProcessor()
    return _audio_processor
