"""
Speech Style Manager

Manages voice configurations for different speaking modes:
- Explain: Clear, neutral, moderate pace
- Teach: Engaging, slightly slower, with natural pauses
"""

import logging
from dataclasses import dataclass
from typing import Optional

from app.core.config import get_settings
from app.core.constants import QUERY_MODE_EXPLAIN, QUERY_MODE_TEACH

logger = logging.getLogger(__name__)


@dataclass
class SpeechStyle:
    """Configuration for a TTS voice style."""
    voice: str
    rate: str           # e.g. "+0%", "-10%", "+15%"
    pitch: str          # e.g. "+0Hz", "-5Hz", "+10Hz"
    volume: str         # e.g. "+0%"
    pause_between_sections: float  # seconds of silence between sections


# ── Default Style Presets ────────────────────────────────────

EXPLAIN_STYLE = SpeechStyle(
    voice="en-US-AriaNeural",
    rate="+0%",
    pitch="+0Hz",
    volume="+0%",
    pause_between_sections=0.8,
)

TEACH_STYLE = SpeechStyle(
    voice="en-US-AriaNeural",
    rate="-8%",      # Slightly slower for teaching
    pitch="+2Hz",    # Slightly higher for engagement
    volume="+0%",
    pause_between_sections=1.2,  # Longer pauses for absorption
)


class SpeechStyleManager:
    """Manages and customises speech styles for TTS."""

    def __init__(self):
        self.settings = get_settings()
        self._styles = {
            QUERY_MODE_EXPLAIN: EXPLAIN_STYLE,
            QUERY_MODE_TEACH: TEACH_STYLE,
        }
        logger.info("SpeechStyleManager initialized")

    def get_style(self, mode: str) -> SpeechStyle:
        """Return the speech style for a given mode."""
        style = self._styles.get(mode, EXPLAIN_STYLE)
        # Override voice from settings if configured
        configured_voice = self.settings.edge_tts_voice
        if configured_voice:
            style = SpeechStyle(
                voice=configured_voice,
                rate=style.rate,
                pitch=style.pitch,
                volume=style.volume,
                pause_between_sections=style.pause_between_sections,
            )
        return style

    def get_custom_style(
        self,
        mode: str,
        voice: Optional[str] = None,
        rate: Optional[str] = None,
        pitch: Optional[str] = None,
    ) -> SpeechStyle:
        """Return a style with optional per-request overrides."""
        base = self.get_style(mode)
        return SpeechStyle(
            voice=voice or base.voice,
            rate=rate or base.rate,
            pitch=pitch or base.pitch,
            volume=base.volume,
            pause_between_sections=base.pause_between_sections,
        )

    def list_available_styles(self) -> dict:
        """Return a summary of available styles."""
        return {
            mode: {
                "voice": style.voice,
                "rate": style.rate,
                "pitch": style.pitch,
                "pause_seconds": style.pause_between_sections,
            }
            for mode, style in self._styles.items()
        }


# ── Module-level factory ─────────────────────────────────────

_style_manager: Optional[SpeechStyleManager] = None


def get_speech_style_manager() -> SpeechStyleManager:
    global _style_manager
    if _style_manager is None:
        _style_manager = SpeechStyleManager()
    return _style_manager
