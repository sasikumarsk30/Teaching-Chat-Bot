"""
Pydantic Request Models

Defines all incoming request schemas for the API endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional
from app.core.constants import (
    QUERY_MODE_EXPLAIN,
    QUERY_MODE_TEACH,
    DEFAULT_CHUNKING_STRATEGY,
    DEFAULT_AUDIO_FORMAT,
)


# ── Document Endpoints ───────────────────────────────────────

class DocumentUploadRequest(BaseModel):
    """Metadata sent alongside a file upload (multipart form)."""
    title: Optional[str] = Field(
        default=None,
        max_length=255,
        description="Optional document title. Defaults to filename.",
    )
    description: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Optional description of the document.",
    )
    tags: Optional[list[str]] = Field(
        default=None,
        description="Optional tags for categorisation.",
    )
    chunking_strategy: str = Field(
        default=DEFAULT_CHUNKING_STRATEGY,
        description="Chunking strategy: semantic | fixed | paragraph.",
    )
    chunk_size: Optional[int] = Field(
        default=None,
        ge=100,
        le=10000,
        description="Override default chunk size (characters).",
    )
    chunk_overlap: Optional[int] = Field(
        default=None,
        ge=0,
        le=2000,
        description="Override default chunk overlap (characters).",
    )
    generate_embeddings: bool = Field(
        default=True,
        description="Whether to generate embeddings immediately after chunking.",
    )


# ── Query / Search Endpoints ────────────────────────────────

class QueryRequest(BaseModel):
    """User query for content retrieval and response generation."""
    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="The user's question or topic to explain/teach.",
    )
    mode: str = Field(
        default=QUERY_MODE_EXPLAIN,
        description="Response mode: 'explain' (simple) or 'teach' (structured lesson).",
    )
    document_id: Optional[str] = Field(
        default=None,
        description="Scope search to a specific document. None = search all.",
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=50,
        description="Number of relevant chunks to retrieve.",
    )
    generate_audio: bool = Field(
        default=False,
        description="Whether to also generate audio for the response.",
    )
    audio_format: str = Field(
        default=DEFAULT_AUDIO_FORMAT,
        description="Audio format when generate_audio=True: mp3 | wav | ogg.",
    )


class ChunkSearchRequest(BaseModel):
    """Direct chunk search (semantic / keyword)."""
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Search query text.",
    )
    document_id: Optional[str] = Field(
        default=None,
        description="Scope to a specific document.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of results to return.",
    )


# ── Audio Endpoints ──────────────────────────────────────────

class AudioGenerationRequest(BaseModel):
    """Request to generate audio from a previously generated response."""
    response_id: str = Field(
        ...,
        description="ID of the generated text response to convert to audio.",
    )
    voice: Optional[str] = Field(
        default=None,
        description="Override default TTS voice.",
    )
    rate: Optional[str] = Field(
        default=None,
        description="Speech rate adjustment, e.g. '+10%' or '-5%'.",
    )
    output_format: str = Field(
        default=DEFAULT_AUDIO_FORMAT,
        description="Audio output format: mp3 | wav | ogg.",
    )
    speaker_wav: Optional[str] = Field(
        default=None,
        description="Path to reference speaker WAV for voice cloning (XTTS v2 / Tortoise only).",
    )
    language: str = Field(
        default="en",
        description="ISO-639-1 language code for multilingual engines.",
    )


class TextToAudioRequest(BaseModel):
    """Direct text-to-audio conversion (bypass retrieval pipeline)."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=50000,
        description="Raw text to convert to speech.",
    )
    mode: str = Field(
        default=QUERY_MODE_EXPLAIN,
        description="Speech style: 'explain' or 'teach'.",
    )
    voice: Optional[str] = Field(default=None)
    rate: Optional[str] = Field(default=None)
    output_format: str = Field(default=DEFAULT_AUDIO_FORMAT)
    speaker_wav: Optional[str] = Field(
        default=None,
        description="Path to reference speaker WAV for voice cloning.",
    )
    language: str = Field(
        default="en",
        description="ISO-639-1 language code for multilingual engines.",
    )


# ── Predefined Content ──────────────────────────────────────

class PredefinedContentRequest(BaseModel):
    """
    Request using predefined selection criteria to extract specific
    content from documents and generate teaching audio.
    """
    topic: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Topic or subject to extract content for.",
    )
    subtopics: Optional[list[str]] = Field(
        default=None,
        description="Specific subtopics to cover.",
    )
    document_ids: Optional[list[str]] = Field(
        default=None,
        description="Limit extraction to specific documents.",
    )
    mode: str = Field(
        default=QUERY_MODE_TEACH,
        description="Response mode: 'explain' or 'teach'.",
    )
    generate_audio: bool = Field(
        default=True,
        description="Generate audio for the response.",
    )
    audio_format: str = Field(default=DEFAULT_AUDIO_FORMAT)
    max_chunks_per_subtopic: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Max chunks to retrieve per subtopic.",
    )
