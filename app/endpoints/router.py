"""
Main Router

Composes all endpoint routers under the /api/v1 prefix.
"""

from fastapi import APIRouter

from app.core.constants import API_V1_PREFIX
from app.endpoints.document_endpoints import router as document_router
from app.endpoints.chunk_endpoints import router as chunk_router
from app.endpoints.query_endpoints import router as query_router
from app.endpoints.audio_endpoints import router as audio_router

api_router = APIRouter(prefix=API_V1_PREFIX)

api_router.include_router(document_router)
api_router.include_router(chunk_router)
api_router.include_router(query_router)
api_router.include_router(audio_router)
