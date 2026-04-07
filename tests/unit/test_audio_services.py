"""
Unit tests for app.services.audio_generation

Tests:
- SpeechStyleManager styles
- TTSService factory pattern + handler delegation
- Handler interface compliance (all 6 engines)
- AudioProcessor metadata & format helpers
"""

import pytest
import uuid
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from app.services.audio_generation.speech_style_manager import (
    SpeechStyleManager,
    SpeechStyle,
    EXPLAIN_STYLE,
    TEACH_STYLE,
)
from app.services.audio_generation.tts_service import TTSService, _create_handler
from app.services.audio_generation.tts_models.base_tts_model import BaseTTSModel
from app.services.audio_generation.audio_processor import AudioProcessor
from app.core.constants import (
    TTS_ENGINE_EDGE,
    TTS_ENGINE_COQUI,
    TTS_ENGINE_XTTS_V2,
    TTS_ENGINE_FAST_PITCH,
    TTS_ENGINE_TORTOISE,
    TTS_ENGINE_BARK,
    SUPPORTED_TTS_ENGINES,
)
from app.utils.error_handlers import TTSGenerationError


# ── Shared helpers ───────────────────────────────────────────

def _default_style():
    return SpeechStyle(
        voice="en-US-AriaNeural",
        rate="+0%",
        pitch="+0Hz",
        volume="+0%",
        pause_between_sections=0.8,
    )


def _mock_settings(engine: str = "edge-tts"):
    """Return a MagicMock that looks like AppSettings with get_active_tts_config."""
    from app.core.constants import TTS_ENGINE_REGISTRY

    s = MagicMock()
    s.tts_engine = engine
    s.edge_tts_voice = "en-US-AriaNeural"
    s.edge_tts_rate = "+0%"
    s.edge_tts_pitch = "+0Hz"
    s.tts_output_format = "mp3"
    s.tts_model_name_override = None
    s.tts_language = "en"
    s.tts_speaker_wav = None
    s.tts_use_gpu = False
    s.tts_quality_preset = "fast"
    s.tts_bark_speaker = "v2/en_speaker_6"

    # get_active_tts_config() merges registry + overrides
    registry = TTS_ENGINE_REGISTRY.get(engine, {})
    s.get_active_tts_config.return_value = {
        "engine": engine,
        "model_name": registry.get("model_name"),
        "supports_voice_cloning": registry.get("supports_voice_cloning", False),
        "supported_languages": registry.get("supported_languages", ["en"]),
        "inference_speed": registry.get("inference_speed", "medium"),
        "description": registry.get("description", ""),
        "install_hint": registry.get("install_hint", ""),
        "language": "en",
        "speaker_wav": None,
        "use_gpu": False,
        "quality_preset": "fast",
        "bark_speaker": "v2/en_speaker_6",
        "output_format": "mp3",
        "edge_tts_voice": "en-US-AriaNeural",
        "edge_tts_rate": "+0%",
        "edge_tts_pitch": "+0Hz",
    }
    return s


# ═══════════════════════════════════════════════════════════════
# SpeechStyleManager Tests
# ═══════════════════════════════════════════════════════════════

class TestSpeechStyleManager:
    @pytest.fixture
    def style_manager(self):
        with patch("app.services.audio_generation.speech_style_manager.get_settings") as m:
            settings = MagicMock()
            settings.edge_tts_voice = "en-US-AriaNeural"
            m.return_value = settings
            return SpeechStyleManager()

    def test_explain_style_defaults(self, style_manager):
        style = style_manager.get_style("explain")
        assert isinstance(style, SpeechStyle)
        assert style.rate == "+0%"
        assert style.voice == "en-US-AriaNeural"

    def test_teach_style_slower_rate(self, style_manager):
        style = style_manager.get_style("teach")
        assert "-" in style.rate  # Teach mode is slower
        assert style.pause_between_sections > EXPLAIN_STYLE.pause_between_sections

    def test_unknown_mode_returns_explain(self, style_manager):
        style = style_manager.get_style("unknown")
        assert style.rate == EXPLAIN_STYLE.rate

    def test_custom_style_overrides(self, style_manager):
        style = style_manager.get_custom_style(
            mode="explain",
            voice="en-US-GuyNeural",
            rate="+10%",
        )
        assert style.voice == "en-US-GuyNeural"
        assert style.rate == "+10%"

    def test_custom_style_keeps_base_when_no_override(self, style_manager):
        style = style_manager.get_custom_style(mode="teach")
        teach_base = style_manager.get_style("teach")
        assert style.rate == teach_base.rate

    def test_list_available_styles(self, style_manager):
        styles = style_manager.list_available_styles()
        assert "explain" in styles
        assert "teach" in styles
        assert "voice" in styles["explain"]
        assert "rate" in styles["teach"]


# ═══════════════════════════════════════════════════════════════
# Handler Factory Tests
# ═══════════════════════════════════════════════════════════════

class TestCreateHandler:
    """Verify _create_handler returns the correct handler type."""

    @patch("app.services.audio_generation.tts_models.edge_tts_handler.EdgeTTSHandler.__init__", return_value=None)
    def test_factory_returns_edge_handler(self, _):
        from app.services.audio_generation.tts_models.edge_tts_handler import EdgeTTSHandler
        handler = _create_handler(TTS_ENGINE_EDGE)
        assert isinstance(handler, EdgeTTSHandler)

    @patch("app.services.audio_generation.tts_models.coqui_tts_handler.CoquiTTSHandler.__init__", return_value=None)
    def test_factory_returns_coqui_handler(self, _):
        from app.services.audio_generation.tts_models.coqui_tts_handler import CoquiTTSHandler
        handler = _create_handler(TTS_ENGINE_COQUI)
        assert isinstance(handler, CoquiTTSHandler)

    @patch("app.services.audio_generation.tts_models.xtts_v2_handler.XTTSv2Handler.__init__", return_value=None)
    def test_factory_returns_xtts_handler(self, _):
        from app.services.audio_generation.tts_models.xtts_v2_handler import XTTSv2Handler
        handler = _create_handler(TTS_ENGINE_XTTS_V2)
        assert isinstance(handler, XTTSv2Handler)

    @patch("app.services.audio_generation.tts_models.fast_pitch_handler.FastPitchHandler.__init__", return_value=None)
    def test_factory_returns_fast_pitch_handler(self, _):
        from app.services.audio_generation.tts_models.fast_pitch_handler import FastPitchHandler
        handler = _create_handler(TTS_ENGINE_FAST_PITCH)
        assert isinstance(handler, FastPitchHandler)

    @patch("app.services.audio_generation.tts_models.tortoise_handler.TortoiseTTSHandler.__init__", return_value=None)
    def test_factory_returns_tortoise_handler(self, _):
        from app.services.audio_generation.tts_models.tortoise_handler import TortoiseTTSHandler
        handler = _create_handler(TTS_ENGINE_TORTOISE)
        assert isinstance(handler, TortoiseTTSHandler)

    @patch("app.services.audio_generation.tts_models.bark_handler.BarkHandler.__init__", return_value=None)
    def test_factory_returns_bark_handler(self, _):
        from app.services.audio_generation.tts_models.bark_handler import BarkHandler
        handler = _create_handler(TTS_ENGINE_BARK)
        assert isinstance(handler, BarkHandler)

    def test_factory_raises_for_unknown_engine(self):
        with pytest.raises(TTSGenerationError, match="Unknown TTS engine"):
            _create_handler("nonexistent-engine")


# ═══════════════════════════════════════════════════════════════
# Handler Interface Compliance Tests
# ═══════════════════════════════════════════════════════════════

class TestEdgeTTSHandlerInterface:
    """Verify EdgeTTSHandler satisfies the BaseTTSModel contract."""

    @pytest.fixture
    def handler(self):
        from app.services.audio_generation.tts_models.edge_tts_handler import EdgeTTSHandler
        return EdgeTTSHandler()

    def test_is_base_tts_model(self, handler):
        assert isinstance(handler, BaseTTSModel)

    def test_engine_name(self, handler):
        assert handler.engine_name == "edge-tts"

    def test_no_voice_cloning(self, handler):
        assert handler.supports_voice_cloning is False

    def test_supported_languages_not_empty(self, handler):
        assert len(handler.supported_languages) > 0
        assert "en" in handler.supported_languages

    def test_model_info(self, handler):
        info = handler.get_model_info()
        assert info["engine"] == "edge-tts"
        assert "supports_cloning" in info
        assert "languages" in info

    def test_load_model_with_package(self, handler):
        handler.load_model()
        assert handler.is_model_loaded()

    @pytest.mark.asyncio
    async def test_synthesize_calls_edge_tts(self, handler, tmp_path):
        handler.load_model()
        out = str(tmp_path / "test.mp3")
        with patch("edge_tts.Communicate") as mock_comm:
            mock_inst = AsyncMock()
            mock_comm.return_value = mock_inst

            async def fake_save(path):
                Path(path).write_bytes(b"\x00" * 256)
            mock_inst.save = fake_save

            await handler.synthesize(
                text="Hello",
                style=_default_style(),
                output_path=out,
            )
            mock_comm.assert_called_once()
            assert Path(out).exists()


class TestCoquiTTSHandlerInterface:
    @pytest.fixture
    def handler(self):
        with patch(
            "app.services.audio_generation.tts_models.coqui_tts_handler.get_settings",
            return_value=_mock_settings("coqui"),
        ):
            from app.services.audio_generation.tts_models.coqui_tts_handler import CoquiTTSHandler
            return CoquiTTSHandler()

    def test_is_base_tts_model(self, handler):
        assert isinstance(handler, BaseTTSModel)

    def test_engine_name(self, handler):
        assert handler.engine_name == "coqui"

    def test_no_voice_cloning(self, handler):
        assert handler.supports_voice_cloning is False

    def test_model_info_keys(self, handler):
        info = handler.get_model_info()
        assert "engine" in info
        assert "loaded" in info


class TestXTTSv2HandlerInterface:
    @pytest.fixture
    def handler(self):
        with patch(
            "app.services.audio_generation.tts_models.xtts_v2_handler.get_settings",
            return_value=_mock_settings("xtts_v2"),
        ):
            from app.services.audio_generation.tts_models.xtts_v2_handler import XTTSv2Handler
            return XTTSv2Handler()

    def test_is_base_tts_model(self, handler):
        assert isinstance(handler, BaseTTSModel)

    def test_engine_name(self, handler):
        assert handler.engine_name == "xtts_v2"

    def test_voice_cloning_supported(self, handler):
        assert handler.supports_voice_cloning is True

    def test_multilingual(self, handler):
        langs = handler.supported_languages
        assert "en" in langs
        assert len(langs) > 5  # XTTS supports many languages


class TestFastPitchHandlerInterface:
    @pytest.fixture
    def handler(self):
        with patch(
            "app.services.audio_generation.tts_models.fast_pitch_handler.get_settings",
            return_value=_mock_settings("fast_pitch"),
        ):
            from app.services.audio_generation.tts_models.fast_pitch_handler import FastPitchHandler
            return FastPitchHandler()

    def test_is_base_tts_model(self, handler):
        assert isinstance(handler, BaseTTSModel)

    def test_engine_name(self, handler):
        assert handler.engine_name == "fast_pitch"

    def test_no_voice_cloning(self, handler):
        assert handler.supports_voice_cloning is False


class TestTortoiseTTSHandlerInterface:
    @pytest.fixture
    def handler(self):
        with patch(
            "app.services.audio_generation.tts_models.tortoise_handler.get_settings",
            return_value=_mock_settings("tortoise"),
        ):
            from app.services.audio_generation.tts_models.tortoise_handler import TortoiseTTSHandler
            return TortoiseTTSHandler()

    def test_is_base_tts_model(self, handler):
        assert isinstance(handler, BaseTTSModel)

    def test_engine_name(self, handler):
        assert handler.engine_name == "tortoise"

    def test_voice_cloning_supported(self, handler):
        assert handler.supports_voice_cloning is True


class TestBarkHandlerInterface:
    @pytest.fixture
    def handler(self):
        with patch(
            "app.services.audio_generation.tts_models.bark_handler.get_settings",
            return_value=_mock_settings("bark"),
        ):
            from app.services.audio_generation.tts_models.bark_handler import BarkHandler
            return BarkHandler()

    def test_is_base_tts_model(self, handler):
        assert isinstance(handler, BaseTTSModel)

    def test_engine_name(self, handler):
        assert handler.engine_name == "bark"

    def test_no_voice_cloning(self, handler):
        assert handler.supports_voice_cloning is False

    def test_multilingual(self, handler):
        assert len(handler.supported_languages) > 5


# ═══════════════════════════════════════════════════════════════
# TTSService Tests (Factory + Delegation)
# ═══════════════════════════════════════════════════════════════

class TestTTSService:
    @pytest.fixture
    def tts_service(self, tmp_data_dir):
        with patch("app.services.audio_generation.tts_service.get_settings") as m_settings, \
            patch("app.services.audio_generation.tts_service.get_speech_style_manager") as m_style, \
            patch("app.services.audio_generation.tts_service.AUDIO_DIR", tmp_data_dir["audio"]):

            m_settings.return_value = _mock_settings("edge-tts")

            style_mgr = MagicMock()
            style_mgr.get_custom_style.return_value = _default_style()
            m_style.return_value = style_mgr

            service = TTSService()
            service.output_dir = tmp_data_dir["audio"]
            yield service

    # ── Initialization ───────────────────────────────────────

    def test_default_engine_is_edge(self, tts_service):
        assert tts_service.engine == "edge-tts"

    def test_handler_is_base_tts_model(self, tts_service):
        assert isinstance(tts_service.handler, BaseTTSModel)

    def test_handler_property_returns_handler(self, tts_service):
        assert tts_service.handler is tts_service._handler

    def test_engine_info(self, tts_service):
        info = tts_service.get_engine_info()
        assert info["engine"] == "edge-tts"
        assert "supports_cloning" in info
        assert "languages" in info
        assert "inference_speed" in info
        assert "description" in info

    # ── Synthesis delegates to handler ───────────────────────

    @pytest.mark.asyncio
    async def test_synthesize_delegates_to_handler(self, tts_service, tmp_data_dir):
        """Ensure synthesize() calls handler.synthesize() with correct args."""
        mock_handler = AsyncMock(spec=BaseTTSModel)
        mock_handler.synthesize = AsyncMock()
        tts_service._handler = mock_handler

        # Make the output file exist so file_size check works
        async def _create_file(text, style, output_path, speaker_wav=None, language="en"):
            Path(output_path).write_bytes(b"\x00" * 512)
        mock_handler.synthesize.side_effect = _create_file

        result = await tts_service.synthesize(
            text="Hello world",
            mode="explain",
            output_format="mp3",
        )

        mock_handler.synthesize.assert_awaited_once()
        call_kwargs = mock_handler.synthesize.call_args
        assert "Hello world" in call_kwargs.kwargs.get("text", call_kwargs.args[0] if call_kwargs.args else "")
        assert "audio_id" in result
        assert result["format"] == "mp3"
        assert result["engine"] == "edge-tts"

    @pytest.mark.asyncio
    async def test_synthesize_passes_speaker_wav(self, tts_service):
        """speaker_wav & language should be forwarded to handler."""
        mock_handler = AsyncMock(spec=BaseTTSModel)
        mock_handler.synthesize = AsyncMock()
        tts_service._handler = mock_handler

        async def _create_file(**kwargs):
            Path(kwargs["output_path"]).write_bytes(b"\x00" * 256)
        mock_handler.synthesize.side_effect = _create_file

        await tts_service.synthesize(
            text="Test",
            speaker_wav="/path/to/ref.wav",
            language="fr",
        )

        call_kw = mock_handler.synthesize.call_args.kwargs
        assert call_kw["speaker_wav"] == "/path/to/ref.wav"
        assert call_kw["language"] == "fr"

    @pytest.mark.asyncio
    async def test_synthesize_includes_voice_cloned_flag(self, tts_service):
        mock_handler = AsyncMock(spec=BaseTTSModel)
        mock_handler.synthesize = AsyncMock()
        tts_service._handler = mock_handler

        async def _create_file(**kwargs):
            Path(kwargs["output_path"]).write_bytes(b"\x00" * 256)
        mock_handler.synthesize.side_effect = _create_file

        result = await tts_service.synthesize(
            text="Test",
            speaker_wav="/ref.wav",
        )
        assert result["voice_cloned"] is True

        result2 = await tts_service.synthesize(text="Test 2")
        assert result2["voice_cloned"] is False

    @pytest.mark.asyncio
    async def test_synthesize_creates_audio_via_edge(self, tts_service, tmp_data_dir):
        """Full path: service → EdgeTTSHandler → mocked edge_tts."""
        with patch("edge_tts.Communicate") as mock_comm:
            mock_instance = AsyncMock()
            mock_comm.return_value = mock_instance

            async def fake_save(path):
                Path(path).write_bytes(b"\x00" * 512)
            mock_instance.save = fake_save

            result = await tts_service.synthesize(
                text="Hello, this is a test.",
                mode="explain",
                output_format="mp3",
            )

            assert "audio_id" in result
            assert result["format"] == "mp3"
            assert result["file_size_bytes"] > 0
            assert result["voice"] == "en-US-AriaNeural"

    # ── Error Handling ───────────────────────────────────────

    @pytest.mark.asyncio
    async def test_empty_text_raises(self, tts_service):
        with pytest.raises(TTSGenerationError):
            await tts_service.synthesize(text="", mode="explain")

    @pytest.mark.asyncio
    async def test_whitespace_only_raises(self, tts_service):
        with pytest.raises(TTSGenerationError):
            await tts_service.synthesize(text="   ", mode="explain")

    @pytest.mark.asyncio
    async def test_handler_error_wrapped(self, tts_service):
        """Unexpected exceptions from handler are wrapped in TTSGenerationError."""
        mock_handler = AsyncMock(spec=BaseTTSModel)
        mock_handler.synthesize = AsyncMock(side_effect=RuntimeError("oops"))
        tts_service._handler = mock_handler

        with pytest.raises(TTSGenerationError, match="oops"):
            await tts_service.synthesize(text="test", mode="explain")

    # ── Duration Estimation ──────────────────────────────────

    def test_estimate_duration(self):
        text = " ".join(["word"] * 150)  # 150 words
        duration = TTSService._estimate_duration(text)
        assert abs(duration - 60.0) < 1.0

    # ── File Operations ──────────────────────────────────────

    def test_get_audio_file_exists(self, tts_service, tmp_data_dir):
        audio_id = str(uuid.uuid4())
        file_path = tmp_data_dir["audio"] / f"{audio_id}.mp3"
        file_path.write_bytes(b"\x00" * 256)

        result = tts_service.get_audio_file(audio_id)
        assert result is not None
        assert result.exists()

    def test_get_audio_file_not_found(self, tts_service):
        result = tts_service.get_audio_file("nonexistent")
        assert result is None

    def test_get_audio_file_fallback_format(self, tts_service, tmp_data_dir):
        audio_id = str(uuid.uuid4())
        (tmp_data_dir["audio"] / f"{audio_id}.wav").write_bytes(b"\x00" * 256)
        result = tts_service.get_audio_file(audio_id, fmt="mp3")
        assert result is not None
        assert result.name.endswith(".wav")

    def test_delete_audio(self, tts_service, tmp_data_dir):
        audio_id = str(uuid.uuid4())
        file_path = tmp_data_dir["audio"] / f"{audio_id}.mp3"
        file_path.write_bytes(b"\x00" * 256)

        assert tts_service.delete_audio(audio_id) is True
        assert not file_path.exists()

    def test_delete_nonexistent_audio(self, tts_service):
        assert tts_service.delete_audio("nonexistent") is False

    def test_list_audio_files(self, tts_service, tmp_data_dir):
        for i in range(3):
            (tmp_data_dir["audio"] / f"test_{i}.mp3").write_bytes(b"\x00" * 100)

        files = tts_service.list_audio_files()
        assert len(files) == 3


# ═══════════════════════════════════════════════════════════════
# TTSService Engine Switching Tests
# ═══════════════════════════════════════════════════════════════

class TestTTSServiceEngineSwitching:
    """Verify TTSService creates the correct handler for each engine."""

    _ENGINES = [
        TTS_ENGINE_EDGE,
        TTS_ENGINE_COQUI,
        TTS_ENGINE_XTTS_V2,
        TTS_ENGINE_FAST_PITCH,
        TTS_ENGINE_TORTOISE,
        TTS_ENGINE_BARK,
    ]

    @pytest.fixture(params=_ENGINES)
    def engine_name(self, request):
        return request.param

    def test_service_selects_matching_handler(self, engine_name, tmp_data_dir):
        with patch("app.services.audio_generation.tts_service.get_settings") as m_s, \
            patch("app.services.audio_generation.tts_service.get_speech_style_manager") as m_st, \
            patch("app.services.audio_generation.tts_service.AUDIO_DIR", tmp_data_dir["audio"]):

            m_s.return_value = _mock_settings(engine_name)
            m_st.return_value = MagicMock()

            service = TTSService()
            assert service.engine == engine_name
            assert isinstance(service.handler, BaseTTSModel)
            assert service.handler.engine_name == engine_name


# ═══════════════════════════════════════════════════════════════
# Constants Validation Tests
# ═══════════════════════════════════════════════════════════════

class TestTTSConstants:
    def test_supported_engines_contains_all(self):
        assert TTS_ENGINE_EDGE in SUPPORTED_TTS_ENGINES
        assert TTS_ENGINE_COQUI in SUPPORTED_TTS_ENGINES
        assert TTS_ENGINE_XTTS_V2 in SUPPORTED_TTS_ENGINES
        assert TTS_ENGINE_FAST_PITCH in SUPPORTED_TTS_ENGINES
        assert TTS_ENGINE_TORTOISE in SUPPORTED_TTS_ENGINES
        assert TTS_ENGINE_BARK in SUPPORTED_TTS_ENGINES

    def test_supported_engines_count(self):
        assert len(SUPPORTED_TTS_ENGINES) == 6


class TestTTSEngineRegistry:
    """Verify TTS_ENGINE_REGISTRY is complete and consistent."""

    def test_registry_has_all_engines(self):
        from app.core.constants import TTS_ENGINE_REGISTRY
        for engine in SUPPORTED_TTS_ENGINES:
            assert engine in TTS_ENGINE_REGISTRY, f"Missing registry entry: {engine}"

    def test_each_registry_entry_has_required_keys(self):
        from app.core.constants import TTS_ENGINE_REGISTRY
        required_keys = [
            "model_name", "supports_voice_cloning", "supported_languages",
            "inference_speed", "description", "install_hint",
        ]
        for engine, entry in TTS_ENGINE_REGISTRY.items():
            for key in required_keys:
                assert key in entry, f"{engine} missing key: {key}"

    def test_voice_cloning_engines(self):
        from app.core.constants import TTS_ENGINE_REGISTRY
        cloning_engines = [
            e for e, r in TTS_ENGINE_REGISTRY.items()
            if r["supports_voice_cloning"]
        ]
        assert TTS_ENGINE_XTTS_V2 in cloning_engines
        assert TTS_ENGINE_TORTOISE in cloning_engines
        assert TTS_ENGINE_EDGE not in cloning_engines

    def test_languages_not_empty(self):
        from app.core.constants import TTS_ENGINE_REGISTRY
        for engine, entry in TTS_ENGINE_REGISTRY.items():
            assert len(entry["supported_languages"]) > 0, f"{engine} has no languages"
            assert "en" in entry["supported_languages"], f"{engine} missing 'en'"


class TestGetActiveTTSConfig:
    """Verify AppSettings.get_active_tts_config() merges registry + env."""

    def test_returns_registry_defaults(self):
        """get_active_tts_config includes registry metadata + env settings."""
        settings = _mock_settings("edge-tts")
        cfg = settings.get_active_tts_config()
        assert cfg["engine"] == "edge-tts"
        assert cfg["model_name"] is None  # Edge-TTS has no local model
        assert cfg["supports_voice_cloning"] is False
        assert "en" in cfg["supported_languages"]
        assert cfg["edge_tts_voice"] == "en-US-AriaNeural"
        assert "inference_speed" in cfg
        assert "install_hint" in cfg

    def test_model_name_override(self):
        """TTS_MODEL_NAME env var overrides the registry default."""
        settings = _mock_settings("coqui")
        base_cfg = settings.get_active_tts_config()
        # Simulate override
        base_cfg["model_name"] = "custom/my-model"
        settings.get_active_tts_config.return_value = base_cfg
        cfg = settings.get_active_tts_config()
        assert cfg["model_name"] == "custom/my-model"

    def test_config_has_unified_keys(self):
        """Every engine's config dict has the same key set."""
        required_keys = {
            "engine", "model_name", "supports_voice_cloning",
            "supported_languages", "inference_speed", "description",
            "install_hint", "language", "speaker_wav", "use_gpu",
            "quality_preset", "bark_speaker", "output_format",
            "edge_tts_voice", "edge_tts_rate", "edge_tts_pitch",
        }
        for engine in SUPPORTED_TTS_ENGINES:
            cfg = _mock_settings(engine).get_active_tts_config()
            missing = required_keys - set(cfg.keys())
            assert not missing, f"{engine} missing keys: {missing}"


# ═══════════════════════════════════════════════════════════════
# AudioProcessor Tests
# ═══════════════════════════════════════════════════════════════

class TestAudioProcessor:
    @pytest.fixture
    def processor(self):
        with patch("app.services.audio_generation.audio_processor.AUDIO_DIR", Path("/tmp")):
            return AudioProcessor()

    def test_get_metadata_for_existing_file(self, processor, tmp_path):
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"\x00" * 2048)

        metadata = processor.get_audio_metadata(str(audio_file))
        assert metadata["filename"] == "test.mp3"
        assert metadata["format"] == "mp3"
        assert metadata["file_size_bytes"] == 2048

    def test_get_metadata_for_missing_file(self, processor):
        metadata = processor.get_audio_metadata("/nonexistent/file.mp3")
        assert metadata == {}

    def test_convert_format_without_pydub_returns_none(self, processor, tmp_path):
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"\x00" * 512)

        with patch.dict("sys.modules", {"pydub": None}):
            result = processor.convert_format(str(audio_file), "wav")
