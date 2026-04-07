"""
Document Endpoints

Handles document upload, listing, detail retrieval, and deletion.
"""

import logging
import time
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from app.models.response_models import (
    DocumentUploadResponse,
    DocumentListResponse,
    DocumentDetailResponse,
    DocumentDeleteResponse,
    DocumentMetadata,
)
from app.services.document_processing.document_ingestion_service import (
    get_document_ingestion_service,
)
from app.services.document_processing.chunking_service import get_chunking_service
from app.services.document_processing.metadata_manager import get_metadata_manager
from app.services.embedding_generation.embedding_service import get_embedding_service
from app.services.embedding_generation.vector_store_service import get_vector_store_service
from app.infrastructure.data_access.staging_manager import get_staging_manager
from app.infrastructure.data_access.parquet_manager import get_parquet_manager
from app.utils.error_handlers import DocumentNotFoundError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(default=None),
    description: Optional[str] = Form(default=None),
    tags: Optional[str] = Form(default=None),
    chunking_strategy: str = Form(default="semantic"),
    chunk_size: Optional[int] = Form(default=None),
    chunk_overlap: Optional[int] = Form(default=None),
    generate_embeddings: bool = Form(default=True),
):
    """
    Upload a document, chunk it, and optionally generate embeddings.

    Supported formats: PDF, DOCX, TXT, MD.
    """
    start_time = time.time()

    ingestion = get_document_ingestion_service()
    chunking = get_chunking_service()
    metadata_mgr = get_metadata_manager()
    staging = get_staging_manager()

    # 1. Read and ingest document
    file_content = await file.read()
    tag_list = [t.strip() for t in tags.split(",")] if tags else None

    doc_data = await ingestion.ingest_document(
        file_content=file_content,
        filename=file.filename or "unnamed",
        title=title,
        description=description,
        tags=tag_list,
    )

    # 2. Chunk the document
    text_content = doc_data.pop("text_content", "")
    doc_data.pop("character_count", None)

    chunks = chunking.chunk_document(
        text=text_content,
        document_id=doc_data["id"],
        strategy=chunking_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # 3. Stage chunks
    if chunks:
        staging.add_chunks(chunks)
        staging.flush()

    # Update chunk count
    metadata_mgr.update_chunk_count(doc_data["id"], len(chunks))

    # 4. Generate embeddings (optional)
    embeddings_count = 0
    if generate_embeddings and chunks:
        embedding_svc = get_embedding_service()
        vector_store = get_vector_store_service()

        texts = [c["content"] for c in chunks]
        embeddings = await embedding_svc.generate_embeddings(texts)
        records = embedding_svc.build_vector_records(chunks, embeddings)
        embeddings_count = vector_store.store_vectors(records)
        metadata_mgr.mark_embeddings_generated(doc_data["id"])

    elapsed_ms = (time.time() - start_time) * 1000

    logger.info(
        f"Document uploaded | id={doc_data['id']} "
        f"chunks={len(chunks)} embeddings={embeddings_count} "
        f"time={elapsed_ms:.0f}ms"
    )

    return DocumentUploadResponse(
        success=True,
        message=f"Document '{doc_data['filename']}' uploaded successfully.",
        processing_time_ms=elapsed_ms,
        document=DocumentMetadata(
            id=doc_data["id"],
            filename=doc_data["filename"],
            title=doc_data["title"],
            description=doc_data.get("description"),
            file_type=doc_data["file_type"],
            file_size_bytes=doc_data["file_size_bytes"],
            total_chunks=len(chunks),
            embeddings_generated=generate_embeddings and embeddings_count > 0,
            tags=doc_data.get("tags"),
            upload_date=doc_data.get("upload_date", datetime.utcnow()),
        ),
        chunks_created=len(chunks),
        embeddings_created=embeddings_count,
    )


@router.get("", response_model=DocumentListResponse)
async def list_documents(page: int = 1, page_size: int = 20):
    """List all uploaded documents with pagination."""
    ingestion = get_document_ingestion_service()
    docs, total = ingestion.list_documents(page, page_size)

    doc_list = []
    for d in docs:
        doc_list.append(DocumentMetadata(
            id=d["id"],
            filename=d["filename"],
            title=d["title"],
            description=d.get("description"),
            file_type=d["file_type"],
            file_size_bytes=d.get("file_size_bytes", 0),
            total_chunks=d.get("total_chunks", 0),
            embeddings_generated=d.get("embeddings_generated", False),
            tags=d.get("tags"),
            upload_date=d.get("upload_date", datetime.utcnow()),
        ))

    return DocumentListResponse(
        success=True,
        documents=doc_list,
        total_count=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{document_id}", response_model=DocumentDetailResponse)
async def get_document(document_id: str):
    """Get detailed information about a specific document."""
    ingestion = get_document_ingestion_service()
    doc = ingestion.get_document(document_id)

    if doc is None:
        raise DocumentNotFoundError(document_id)

    return DocumentDetailResponse(
        success=True,
        document=DocumentMetadata(
            id=doc["id"],
            filename=doc["filename"],
            title=doc["title"],
            description=doc.get("description"),
            file_type=doc["file_type"],
            file_size_bytes=doc.get("file_size_bytes", 0),
            total_chunks=doc.get("total_chunks", 0),
            embeddings_generated=doc.get("embeddings_generated", False),
            tags=doc.get("tags"),
            upload_date=doc.get("upload_date", datetime.utcnow()),
        ),
    )


@router.delete("/{document_id}", response_model=DocumentDeleteResponse)
async def delete_document(document_id: str):
    """Delete a document and all associated chunks, vectors, and audio."""
    ingestion = get_document_ingestion_service()
    metadata_mgr = get_metadata_manager()
    parquet = get_parquet_manager()

    doc = ingestion.get_document(document_id)
    if doc is None:
        raise DocumentNotFoundError(document_id)

    # Delete chunks from DuckDB
    chunks_deleted = metadata_mgr.delete_chunks_for_document(document_id)

    # Delete from Parquet
    parquet.delete_document_chunks(document_id)

    # Delete document file and metadata
    ingestion.delete_document(document_id)

    return DocumentDeleteResponse(
        success=True,
        message=f"Document '{document_id}' deleted successfully.",
        document_id=document_id,
        chunks_deleted=chunks_deleted,
        audio_files_deleted=0,
    )
