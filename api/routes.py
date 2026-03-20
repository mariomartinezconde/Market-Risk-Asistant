"""
FastAPI Routes
==============
Endpoints:
  GET  /           → redirect to /docs
  GET  /health     → system health check
  POST /query      → main RAG query endpoint
  POST /ingest     → (admin) trigger document ingestion
  GET  /documents  → list indexed documents
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import RedirectResponse

from app.logger import get_logger
from app.models import (
    DocumentType,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from app.orchestrator import answer_query
from config import get_settings
from retrieval.ingestion import ingest_directory
from retrieval.vector_store import get_vector_store

logger = get_logger(__name__)
router = APIRouter()


# ── Root ─────────────────────────────────────────────────────────────────────

@router.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


# ── Health ────────────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System health check",
    tags=["Operations"],
)
async def health():
    settings = get_settings()
    store = get_vector_store()
    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        index_loaded=store.is_loaded,
        total_chunks_indexed=store.total_chunks,
        model=settings.CLAUDE_MODEL,
    )


# ── Query ─────────────────────────────────────────────────────────────────────

@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Query the Market Risk AI Assistant",
    tags=["RAG"],
    responses={
        200: {"description": "Structured answer with citations and governance flags."},
        503: {"description": "Vector store not loaded. Run /ingest first."},
    },
)
async def query(request: QueryRequest):
    """
    Submit a natural-language question about market risk regulations.

    The response includes:
    - **answer**: grounded, cited response.
    - **sources**: list of document chunks used.
    - **confidence**: HIGH / MEDIUM / LOW / INSUFFICIENT.
    - **conflict_detected**: whether retrieved chunks contradict each other.
    - **escalation_recommended**: whether human review is advised.
    """
    store = get_vector_store()
    if not store.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "The vector index is not loaded. "
                "POST /ingest to build the index first, or ensure DATA_DIR contains documents."
            ),
        )

    logger.info("routes.query_received", query=request.query[:80])
    try:
        response = answer_query(request)
    except Exception as exc:
        logger.error("routes.query_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(exc)}",
        )
    return response


# ── Ingest ────────────────────────────────────────────────────────────────────

@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="(Admin) Ingest documents and rebuild vector index",
    tags=["Operations"],
)
async def ingest(request: IngestRequest | None = None):
    """
    Trigger the document ingestion pipeline.

    Reads all PDF, DOCX, and TXT files from DATA_DIR (or the supplied
    `directory`), embeds them, and rebuilds the FAISS index.

    **This endpoint should be protected in production** (set API_KEY env var).
    """
    settings = get_settings()
    directory = (request.directory if request else None) or settings.DATA_DIR

    logger.info("routes.ingest_start", directory=directory)
    errors: list[str] = []
    chunks = []

    try:
        chunks = list(ingest_directory(directory, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP))
    except Exception as exc:
        errors.append(str(exc))
        logger.error("routes.ingest_failed", error=str(exc))

    if not chunks:
        return IngestResponse(
            status="warning",
            documents_processed=0,
            chunks_created=0,
            errors=errors or ["No documents found or all extractions failed."],
        )

    store = get_vector_store()
    try:
        store.build(chunks)
    except Exception as exc:
        errors.append(f"Index build error: {exc}")
        return IngestResponse(
            status="error",
            documents_processed=0,
            chunks_created=0,
            errors=errors,
        )

    doc_names = {c.metadata.doc_name for c in chunks}
    logger.info("routes.ingest_done", docs=len(doc_names), chunks=len(chunks))
    return IngestResponse(
        status="ok",
        documents_processed=len(doc_names),
        chunks_created=len(chunks),
        errors=errors,
    )


# ── Documents ─────────────────────────────────────────────────────────────────

@router.get(
    "/documents",
    summary="List all indexed documents",
    tags=["Operations"],
)
async def list_documents():
    """
    Returns a summary of every document currently in the vector index.
    Useful for verifying the corpus before querying.
    """
    store = get_vector_store()
    if not store.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector index not loaded.",
        )

    seen: dict[str, dict] = {}
    for chunk in store._chunks:
        m = chunk.metadata
        if m.doc_name not in seen:
            seen[m.doc_name] = {
                "doc_name": m.doc_name,
                "doc_type": m.doc_type.value,
                "version": m.version,
                "owner": m.owner,
                "source_file": m.source_file,
                "chunk_count": 0,
            }
        seen[m.doc_name]["chunk_count"] += 1

    return {"total_documents": len(seen), "documents": list(seen.values())}
