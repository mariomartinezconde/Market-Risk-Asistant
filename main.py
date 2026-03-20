"""
Market Risk AI Assistant — Application Entry Point
===================================================
Start with:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
Or via the helper script:
    python main.py
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.middleware import api_key_middleware, logging_middleware
from api.routes import router
from app.logger import configure_logging, get_logger
from config import get_settings
from retrieval.vector_store import get_vector_store
from retrieval.ingestion import ingest_directory

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: configure logging, attempt to load existing vector index.
    If no index exists on disk, run ingestion from DATA_DIR automatically.
    Shutdown: flush logs.
    """
    settings = get_settings()
    configure_logging("DEBUG" if settings.DEBUG else "INFO")

    logger.info("startup.begin", app=settings.APP_NAME, env=settings.APP_ENV)

    store = get_vector_store()
    loaded = store.load()

    if not loaded:
        logger.info("startup.no_index_found_running_ingestion", data_dir=settings.DATA_DIR)
        try:
            chunks = list(
                ingest_directory(settings.DATA_DIR, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
            )
            if chunks:
                store.build(chunks)
                logger.info("startup.ingestion_complete", chunks=len(chunks))
            else:
                logger.warning(
                    "startup.no_documents_found",
                    data_dir=settings.DATA_DIR,
                    hint="Place documents in DATA_DIR and POST /ingest, or set DATA_DIR env var.",
                )
        except Exception as exc:
            logger.error("startup.ingestion_failed", error=str(exc))

    logger.info(
        "startup.ready",
        index_loaded=store.is_loaded,
        chunks=store.total_chunks,
    )
    yield
    logger.info("shutdown.graceful")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "Production-grade RAG system for market risk regulation queries. "
            "Grounded answers, mandatory citations, conflict detection, audit logging."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Custom middleware (order matters: registered last = executed first)
    app.middleware("http")(api_key_middleware)
    app.middleware("http")(logging_middleware)

    app.include_router(router)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level="debug" if settings.DEBUG else "info",
    )
