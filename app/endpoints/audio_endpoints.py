"""
Audio Endpoints

Handles audio generation, retrieval, and management.
"""

import logging
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

from app.models.request_models import AudioGenerationRequest, TextToAudioRequest
from app.models.response_models import (
    AudioGenerationResponse,
    AudioMetadata,
    AudioListResponse,
    BaseResponse,
)
from app.services.audio_generation.tts_service import get_tts_service
from app.services.audio_generation.audio_processor import get_audio_processor
from app.infrastructure.cache.response_cache import get_response_cache
from app.utils.error_handlers import AudioNotFoundError, TTSGenerationError
from app.utils.validators import validate_audio_format

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/audio", tags=["Audio"])


@router.post("/generate", response_model=AudioGenerationResponse)
async def generate_audio_from_response(request: AudioGenerationRequest):
    """
    Generate audio from a previously cached text response.
    """
    cache = get_response_cache()
    tts = get_tts_service()

    # Fetch the cached response
    cached = cache.get_by_id(request.response_id)
    if cached is None:
        raise AudioNotFoundError(request.response_id)

    fmt = validate_audio_format(request.output_format)

    result = await tts.synthesize(
        text=cached["response_text"],
        mode=cached.get("mode", "explain"),
        voice=request.voice,
        rate=request.rate,
        output_format=fmt,
        speaker_wav=request.speaker_wav,
        language=request.language,
    )

    # Update cache with audio path
    cache.update_audio(
        request.response_id,
        result["file_path"],
        result.get("duration_seconds"),
    )

    from datetime import datetime

    return AudioGenerationResponse(
        success=True,
        message="Audio generated successfully.",
        audio=AudioMetadata(
            audio_id=result["audio_id"],
            response_id=request.response_id,
            file_path=result["file_path"],
            format=result["format"],
            duration_seconds=result.get("duration_seconds"),
            file_size_bytes=result.get("file_size_bytes"),
            voice=result["voice"],
            created_at=datetime.utcnow(),
        ),
        download_url=f"/api/v1/audio/{result['audio_id']}/download",
    )


@router.post("/synthesize", response_model=AudioGenerationResponse)
async def synthesize_text(request: TextToAudioRequest):
    """
    Direct text-to-audio conversion (bypass document retrieval).
    """
    tts = get_tts_service()
    fmt = validate_audio_format(request.output_format)

    result = await tts.synthesize(
        text=request.text,
        mode=request.mode,
        voice=request.voice,
        rate=request.rate,
        output_format=fmt,
        speaker_wav=request.speaker_wav,
        language=request.language,
    )

    from datetime import datetime

    return AudioGenerationResponse(
        success=True,
        message="Audio synthesized successfully.",
        audio=AudioMetadata(
            audio_id=result["audio_id"],
            response_id="direct",
            file_path=result["file_path"],
            format=result["format"],
            duration_seconds=result.get("duration_seconds"),
            file_size_bytes=result.get("file_size_bytes"),
            voice=result["voice"],
            created_at=datetime.utcnow(),
        ),
        download_url=f"/api/v1/audio/{result['audio_id']}/download",
    )


@router.get("/{audio_id}/download")
async def download_audio(audio_id: str):
    """Download a generated audio file."""
    tts = get_tts_service()
    file_path = tts.get_audio_file(audio_id)

    if file_path is None or not file_path.exists():
        raise AudioNotFoundError(audio_id)

    media_types = {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".ogg": "audio/ogg",
    }
    media_type = media_types.get(file_path.suffix, "application/octet-stream")

    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=file_path.name,
    )


@router.get("/{audio_id}/metadata")
async def get_audio_metadata(audio_id: str):
    """Get metadata for a generated audio file."""
    tts = get_tts_service()
    processor = get_audio_processor()

    file_path = tts.get_audio_file(audio_id)
    if file_path is None:
        raise AudioNotFoundError(audio_id)

    metadata = processor.get_audio_metadata(str(file_path))
    return {"success": True, "metadata": metadata}


@router.get("", response_model=AudioListResponse)
async def list_audio_files():
    """List all generated audio files."""
    tts = get_tts_service()
    files = tts.list_audio_files()

    from datetime import datetime

    audio_list = [
        AudioMetadata(
            audio_id=f["audio_id"],
            response_id="",
            file_path="",
            format=f["format"],
            file_size_bytes=f.get("file_size_bytes"),
            voice="",
            created_at=datetime.utcnow(),
        )
        for f in files
    ]

    return AudioListResponse(
        success=True,
        audio_files=audio_list,
        total_count=len(audio_list),
    )


@router.delete("/{audio_id}", response_model=BaseResponse)
async def delete_audio(audio_id: str):
    """Delete a generated audio file."""
    tts = get_tts_service()
    deleted = tts.delete_audio(audio_id)

    if not deleted:
        raise AudioNotFoundError(audio_id)

    return BaseResponse(
        success=True,
        message=f"Audio '{audio_id}' deleted successfully.",
    )
