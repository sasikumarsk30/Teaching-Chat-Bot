"""
Query Endpoints

Handles query/explain/teach requests.
Combines content retrieval, LLM response generation,
and optional audio synthesis.
"""

import logging
import time

from fastapi import APIRouter

from app.models.request_models import (
    QueryRequest,
    PredefinedContentRequest,
)
from app.models.response_models import (
    QueryResponse,
    PredefinedContentResponse,
    SourceChunk,
)
from app.services.content_analysis.response_generator import get_response_generator
from app.services.audio_generation.tts_service import get_tts_service
from app.infrastructure.cache.response_cache import get_response_cache
from app.utils.validators import validate_query_mode

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["Query"])


@router.post("", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Process a user query: retrieve relevant content, generate a response,
    and optionally convert to audio.
    """
    start_time = time.time()

    mode = validate_query_mode(request.mode)
    generator = get_response_generator()

    # Generate response
    result = await generator.generate(
        question=request.question,
        mode=mode,
        document_id=request.document_id,
        top_k=request.top_k,
    )

    # Optionally generate audio
    audio_url = None
    audio_duration = None

    if request.generate_audio and result.get("response_text"):
        tts = get_tts_service()
        audio_result = await tts.synthesize(
            text=result["response_text"],
            mode=mode,
            output_format=request.audio_format,
        )
        audio_url = f"/api/v1/audio/{audio_result['audio_id']}"
        audio_duration = audio_result.get("duration_seconds")

        # Update cache with audio info
        cache = get_response_cache()
        if result.get("response_id"):
            cache.update_audio(
                result["response_id"],
                audio_result["file_path"],
                audio_duration,
            )

    elapsed_ms = (time.time() - start_time) * 1000

    source_chunks = [
        SourceChunk(**c) for c in result.get("source_chunks", [])
    ]

    return QueryResponse(
        success=True,
        message="Response generated successfully.",
        processing_time_ms=elapsed_ms,
        response_id=result.get("response_id", ""),
        question=request.question,
        mode=mode,
        response_text=result.get("response_text", ""),
        source_chunks=source_chunks,
        chunks_retrieved=len(source_chunks),
        audio_url=audio_url,
        audio_duration_seconds=audio_duration,
    )


@router.post("/explain", response_model=QueryResponse)
async def explain(request: QueryRequest):
    """Shortcut endpoint that forces 'explain' mode."""
    request.mode = "explain"
    return await query(request)


@router.post("/teach", response_model=QueryResponse)
async def teach(request: QueryRequest):
    """Shortcut endpoint that forces 'teach' mode."""
    request.mode = "teach"
    return await query(request)


@router.post("/predefined", response_model=PredefinedContentResponse)
async def predefined_content(request: PredefinedContentRequest):
    """
    Generate a structured teaching response for a predefined topic
    with specific subtopics.
    """
    start_time = time.time()

    mode = validate_query_mode(request.mode)
    generator = get_response_generator()

    subtopics = request.subtopics or [request.topic]

    result = await generator.generate_predefined(
        topic=request.topic,
        subtopics=subtopics,
        mode=mode,
        document_ids=request.document_ids,
        max_chunks_per_subtopic=request.max_chunks_per_subtopic,
    )

    # Optionally generate audio
    audio_url = None
    audio_duration = None

    if request.generate_audio and result.get("response_text"):
        tts = get_tts_service()
        audio_result = await tts.synthesize(
            text=result["response_text"],
            mode=mode,
            output_format=request.audio_format,
        )
        audio_url = f"/api/v1/audio/{audio_result['audio_id']}"
        audio_duration = audio_result.get("duration_seconds")

    elapsed_ms = (time.time() - start_time) * 1000

    source_chunks = [
        SourceChunk(**c) for c in result.get("source_chunks", [])
    ]

    return PredefinedContentResponse(
        success=True,
        message="Predefined content generated successfully.",
        processing_time_ms=elapsed_ms,
        response_id=result.get("response_id", ""),
        topic=request.topic,
        subtopics_covered=result.get("subtopics_covered", []),
        response_text=result.get("response_text", ""),
        source_chunks=source_chunks,
        audio_url=audio_url,
        audio_duration_seconds=audio_duration,
    )
