"""
Document Audio Generation Service - Main Application

FastAPI application entry point with lifespan management.
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.endpoints.router import api_router
from app.infrastructure.data_access.duckdb_manager import get_duckdb_manager
from app.infrastructure.data_access.staging_manager import get_staging_manager
from app.models.response_models import HealthCheckResponse
from app.utils.error_handlers import AppError, app_error_handler

logger = logging.getLogger(__name__)


# ── Lifespan Manager ─────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    settings = get_settings()
    start = time.time()

    # ── Startup ──────────────────────────────────────────────
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info(
        f"Starting {settings.app_name} v{settings.app_version} "
        f"[{settings.app_environment}]"
    )

    # Initialize DuckDB schema
    db = get_duckdb_manager()
    db.initialize_schema()
    logger.info("DuckDB schema initialized")

    elapsed = (time.time() - start) * 1000
    logger.info(f"Startup complete in {elapsed:.0f}ms")

    yield

    # ── Shutdown ─────────────────────────────────────────────
    logger.info("Shutting down...")

    staging = get_staging_manager()
    staging.shutdown()

    db.close()
    logger.info("Shutdown complete")


# ── Create Application ───────────────────────────────────────

def create_app() -> FastAPI:
    """Application factory."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "A standalone service for processing documents, generating "
            "educational explanations, and converting them to audio. "
            "Supports 'explain' (simple) and 'teach' (structured lesson) modes."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS ─────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Exception Handlers ───────────────────────────────────
    app.add_exception_handler(AppError, app_error_handler)

    # ── Routers ──────────────────────────────────────────────
    app.include_router(api_router)

    # ── Health Check ─────────────────────────────────────────
    @app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
    async def health_check():
        db = get_duckdb_manager()
        return HealthCheckResponse(
            status="healthy",
            version=settings.app_version,
            environment=settings.app_environment,
            services={
                "duckdb": "connected" if db.connection else "disconnected",
                "tts_engine": settings.tts_engine,
                "llm_provider": settings.llm_provider,
                "embedding_model": settings.embedding_model_name,
            },
        )

    @app.get("/", tags=["Root"])
    async def root():
        return {
            "service": settings.app_name,
            "version": settings.app_version,
            "docs": "/docs",
            "health": "/health",
        }

    return app


# ── Application Instance ─────────────────────────────────────

app = create_app()
