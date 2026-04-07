"""
TTS Models Package

Provides a pluggable architecture for multiple TTS engines.
Each handler inherits from BaseTTSModel and is selected via configuration.
"""

from app.services.audio_generation.tts_models.base_tts_model import BaseTTSModel

__all__ = ["BaseTTSModel"]
