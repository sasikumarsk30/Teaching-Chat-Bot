"""
Pydantic Response Models

Defines all outgoing response schemas for the API endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional, Any
from datetime import datetime


# ── Generic Wrapper ──────────────────────────────────────────

class BaseResponse(BaseModel):
    """Standard envelope for every API response."""
    success: bool = True
    message: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    processing_time_ms: Optional[float] = None


# ── Document Responses ───────────────────────────────────────

class DocumentMetadata(BaseModel):
    """Metadata for a single document."""
    id: str
    filename: str
    title: str
    description: Optional[str] = None
    file_type: str
    file_size_bytes: int
    total_chunks: int = 0
    embeddings_generated: bool = False
    tags: Optional[list[str]] = None
    upload_date: datetime
    metadata: Optional[dict[str, Any]] = None


class DocumentUploadResponse(BaseResponse):
    """Response after uploading a document."""
    document: Optional[DocumentMetadata] = None
    chunks_created: int = 0
    embeddings_created: int = 0


class DocumentListResponse(BaseResponse):
    """Paginated list of documents."""
    documents: list[DocumentMetadata] = []
    total_count: int = 0
    page: int = 1
    page_size: int = 20


class DocumentDetailResponse(BaseResponse):
    """Detailed document data including chunk summary."""
    document: Optional[DocumentMetadata] = None


class DocumentDeleteResponse(BaseResponse):
    """Confirmation of document deletion."""
    document_id: str = ""
    chunks_deleted: int = 0
    audio_files_deleted: int = 0


# ── Chunk Responses ──────────────────────────────────────────

class ChunkData(BaseModel):
    """Single chunk representation."""
    id: str
    document_id: str
    sequence: int
    content: str
    chunk_size: int
    start_char: int
    end_char: int
    metadata: Optional[dict[str, Any]] = None
    similarity_score: Optional[float] = None


class ChunkListResponse(BaseResponse):
    """List of chunks for a document."""
    chunks: list[ChunkData] = []
    total_count: int = 0
    document_id: Optional[str] = None


class ChunkSearchResponse(BaseResponse):
    """Search results over chunks."""
    query: str = ""
    results: list[ChunkData] = []
    total_results: int = 0


# ── Query / Content Responses ────────────────────────────────

class SourceChunk(BaseModel):
    """A source chunk referenced in a generated response."""
    chunk_id: str
    document_id: str
    document_title: str
    sequence: int
    similarity_score: float
    snippet: str = Field(
        ..., description="Preview of the chunk content (first 200 chars)."
    )


class QueryResponse(BaseResponse):
    """Response from query/explain/teach endpoints."""
    response_id: str = ""
    question: str = ""
    mode: str = ""
    response_text: str = ""
    source_chunks: list[SourceChunk] = []
    chunks_retrieved: int = 0
    audio_url: Optional[str] = None
    audio_duration_seconds: Optional[float] = None


class PredefinedContentResponse(BaseResponse):
    """Response from predefined content extraction."""
    response_id: str = ""
    topic: str = ""
    subtopics_covered: list[str] = []
    response_text: str = ""
    source_chunks: list[SourceChunk] = []
    audio_url: Optional[str] = None
    audio_duration_seconds: Optional[float] = None


# ── Audio Responses ──────────────────────────────────────────

class AudioMetadata(BaseModel):
    """Metadata for a generated audio file."""
    audio_id: str
    response_id: str
    file_path: str
    format: str
    duration_seconds: Optional[float] = None
    file_size_bytes: Optional[int] = None
    voice: str
    created_at: datetime


class AudioGenerationResponse(BaseResponse):
    """Response after audio generation."""
    audio: Optional[AudioMetadata] = None
    download_url: str = ""


class AudioListResponse(BaseResponse):
    """List of generated audio files."""
    audio_files: list[AudioMetadata] = []
    total_count: int = 0


# ── Health Check ─────────────────────────────────────────────

class HealthCheckResponse(BaseModel):
    """Health check status."""
    status: str = "healthy"
    version: str = ""
    environment: str = ""
    services: dict[str, str] = {}
