"""System health and admin status routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.database import get_db
from app.models.db_models import ConversationModel, DocumentModel
from app.schemas.system import AdminStatusResponse, HealthResponse
from app.vectorstore.store import get_store

router = APIRouter(tags=["System"])


@router.get("/health", response_model=HealthResponse)
async def health():
    s = get_settings()
    return HealthResponse(status="ok", version=s.APP_VERSION, environment=s.APP_ENV)


@router.get("/admin/status", response_model=AdminStatusResponse)
async def admin_status(db: AsyncSession = Depends(get_db)):
    s = get_settings()
    store = get_store()

    doc_count = (await db.execute(select(func.count()).select_from(DocumentModel))).scalar() or 0
    conv_count = (await db.execute(select(func.count()).select_from(ConversationModel))).scalar() or 0

    return AdminStatusResponse(
        status="ok",
        version=s.APP_VERSION,
        llm_provider=s.LLM_PROVIDER,
        llm_model=s.effective_model_name,
        llm_configured=s.llm_configured,
        vectorstore_loaded=store.is_loaded,
        total_chunks=store.total_chunks,
        total_documents=doc_count,
        total_conversations=conv_count,
        embedding_model=s.EMBEDDING_MODEL_NAME,
    )
