"""
Chunk Endpoints

Handles chunk listing, search, and individual chunk retrieval.
"""

import json
import logging
import time

from fastapi import APIRouter

from app.models.request_models import ChunkSearchRequest
from app.models.response_models import (
    ChunkListResponse,
    ChunkSearchResponse,
    ChunkData,
)
from app.infrastructure.data_access.duckdb_manager import get_duckdb_manager
from app.services.embedding_generation.embedding_service import get_embedding_service
from app.services.embedding_generation.similarity_search import get_similarity_search
from app.core.constants import TABLE_CHUNKS
from app.utils.error_handlers import ChunkNotFoundError, DocumentNotFoundError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chunks", tags=["Chunks"])


@router.get("", response_model=ChunkListResponse)
async def list_chunks(doc_id: str | None = None, page: int = 1, page_size: int = 20):
    """
    List chunks, optionally filtered by document ID.
    """
    db = get_duckdb_manager()
    offset = (page - 1) * page_size

    if doc_id:
        rows = db.fetch_all(
            f"SELECT * FROM {TABLE_CHUNKS} WHERE document_id = ? "
            f"ORDER BY sequence LIMIT ? OFFSET ?",
            [doc_id, page_size, offset],
        )
        total_result = db.fetch_one(
            f"SELECT COUNT(*) as cnt FROM {TABLE_CHUNKS} WHERE document_id = ?",
            [doc_id],
        )
    else:
        rows = db.fetch_all(
            f"SELECT * FROM {TABLE_CHUNKS} ORDER BY document_id, sequence "
            f"LIMIT ? OFFSET ?",
            [page_size, offset],
        )
        total_result = db.fetch_one(
            f"SELECT COUNT(*) as cnt FROM {TABLE_CHUNKS}"
        )

    total = total_result["cnt"] if total_result else 0

    chunk_list = [
        ChunkData(
            id=r["id"],
            document_id=r["document_id"],
            sequence=r["sequence"],
            content=r["content"],
            chunk_size=r.get("chunk_size", len(r["content"])),
            start_char=r.get("start_char", 0),
            end_char=r.get("end_char", 0),
            metadata=json.loads(r.get("metadata")) if r.get("metadata") else None,
        )
        for r in rows
    ]

    return ChunkListResponse(
        success=True,
        chunks=chunk_list,
        total_count=total,
        document_id=doc_id,
    )


@router.get("/{chunk_id}")
async def get_chunk(chunk_id: str):
    """Retrieve a specific chunk by ID."""
    db = get_duckdb_manager()
    row = db.fetch_one(
        f"SELECT * FROM {TABLE_CHUNKS} WHERE id = ?", [chunk_id]
    )

    if row is None:
        raise ChunkNotFoundError(chunk_id)

    return {
        "success": True,
        "chunk": ChunkData(
            id=row["id"],
            document_id=row["document_id"],
            sequence=row["sequence"],
            content=row["content"],
            chunk_size=row.get("chunk_size", len(row["content"])),
            start_char=row.get("start_char", 0),
            end_char=row.get("end_char", 0),
            metadata=row.get("metadata"),
        ),
    }


@router.post("/search", response_model=ChunkSearchResponse)
async def search_chunks(request: ChunkSearchRequest):
    """
    Semantic search across all chunks.
    Returns the most relevant chunks ranked by similarity.
    """
    start_time = time.time()

    embedding_svc = get_embedding_service()
    similarity = get_similarity_search()

    # Generate query embedding
    query_embedding = await embedding_svc.generate_query_embedding(request.query)

    # Search
    results = similarity.search(
        query_embedding=query_embedding,
        top_k=request.top_k,
        document_id=request.document_id,
    )

    elapsed_ms = (time.time() - start_time) * 1000

    chunk_results = [
        ChunkData(
            id=r.get("chunk_id", ""),
            document_id=r.get("document_id", ""),
            sequence=r.get("sequence", 0),
            content=r.get("content", ""),
            chunk_size=len(r.get("content", "")),
            start_char=0,
            end_char=0,
            similarity_score=r.get("similarity_score"),
        )
        for r in results
    ]

    return ChunkSearchResponse(
        success=True,
        query=request.query,
        results=chunk_results,
        total_results=len(chunk_results),
        processing_time_ms=elapsed_ms,
    )
