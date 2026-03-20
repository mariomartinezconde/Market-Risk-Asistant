"""Document management API routes."""
from __future__ import annotations

import asyncio
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.schemas.documents import (
    DeleteResponse, DocumentOut, ReindexAllResponse,
    ReindexResponse, UploadResponse,
)
from app.services.document_service import (
    DocumentServiceError, delete_document, get_document,
    ingest_document, list_documents, upload_document,
)

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload(file: UploadFile = File(...), db: AsyncSession = Depends(get_db)):
    content = await file.read()
    try:
        doc = await upload_document(content, file.filename, db)
    except DocumentServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Kick off ingestion in background
    asyncio.create_task(_ingest_bg(doc.id, db))

    return UploadResponse(
        document_id=doc.id,
        filename=doc.original_filename,
        status="uploaded",
        message="Document uploaded. Indexing started in background.",
    )


@router.get("", response_model=list[DocumentOut])
async def get_documents(db: AsyncSession = Depends(get_db)):
    return await list_documents(db)


@router.get("/{document_id}", response_model=DocumentOut)
async def get_document_by_id(document_id: str, db: AsyncSession = Depends(get_db)):
    doc = await get_document(document_id, db)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    return doc


@router.delete("/{document_id}", response_model=DeleteResponse)
async def delete_doc(document_id: str, db: AsyncSession = Depends(get_db)):
    deleted = await delete_document(document_id, db)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found.")
    return DeleteResponse(
        document_id=document_id, deleted=True, message="Document deleted."
    )


@router.post("/{document_id}/reindex", response_model=ReindexResponse)
async def reindex_one(document_id: str, db: AsyncSession = Depends(get_db)):
    try:
        return await ingest_document(document_id, db)
    except DocumentServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/reindex-all", response_model=ReindexAllResponse)
async def reindex_all(db: AsyncSession = Depends(get_db)):
    docs = await list_documents(db)
    results, succeeded, failed = [], 0, 0
    for doc in docs:
        try:
            r = await ingest_document(doc.id, db)
            results.append(r)
            succeeded += 1
        except DocumentServiceError as e:
            results.append(ReindexResponse(
                document_id=doc.id, status="failed", chunks_created=0, message=str(e)
            ))
            failed += 1
    return ReindexAllResponse(
        total=len(docs), succeeded=succeeded, failed=failed, results=results
    )


async def _ingest_bg(doc_id: str, db: AsyncSession):
    """Background ingestion — called from create_task."""
    from app.db.database import get_session_factory
    async with get_session_factory()() as session:
        try:
            await ingest_document(doc_id, session)
        except Exception:
            pass
