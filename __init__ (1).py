"""Pydantic schemas for chat endpoints."""
from __future__ import annotations

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class SourceCitation(BaseModel):
    doc_id: str
    doc_name: str
    doc_type: str
    version: Optional[str]
    section_title: Optional[str]
    excerpt: str
    similarity_score: float


class MessageOut(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str
    sources: list[SourceCitation] = []
    confidence: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class ConversationOut(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    messages: list[MessageOut] = []

    class Config:
        from_attributes = True


class ConversationSummary(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0

    class Config:
        from_attributes = True


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    filter_doc_type: Optional[str] = None
    filter_doc_id: Optional[str] = None


class ChatResponse(BaseModel):
    message: MessageOut
    conversation_id: str


class CreateConversationRequest(BaseModel):
    title: Optional[str] = "New conversation"
