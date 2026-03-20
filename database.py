"""Pydantic schemas for document endpoints."""
from __future__ import annotations

from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class DocumentOut(BaseModel):
    id: str
    filename: str
    original_filename: str
    file_type: str
    file_size_bytes: int
    doc_type: str
    version: Optional[str]
    status: str
    chunk_count: int
    error_message: Optional[str]
    uploaded_at: datetime
    indexed_at: Optional[datetime]

    class Config:
        from_attributes = True


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    message: str


class ReindexResponse(BaseModel):
    document_id: str
    status: str
    chunks_created: int
    message: str


class ReindexAllResponse(BaseModel):
    total: int
    succeeded: int
    failed: int
    results: list[ReindexResponse]


class DeleteResponse(BaseModel):
    document_id: str
    deleted: bool
    message: str
