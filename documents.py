"""System / health schemas."""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str


class AdminStatusResponse(BaseModel):
    status: str
    version: str
    llm_provider: str
    llm_model: str
    llm_configured: bool
    vectorstore_loaded: bool
    total_chunks: int
    total_documents: int
    total_conversations: int
    embedding_model: str
