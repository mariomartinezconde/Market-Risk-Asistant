"""Chat API routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.schemas.chat import (
    ChatRequest, ChatResponse, ConversationOut,
    ConversationSummary, CreateConversationRequest,
)
from app.services.chat_service import (
    create_conversation, delete_conversation,
    get_conversation, list_conversations, post_message,
)

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/conversations", response_model=ConversationOut, status_code=201)
async def new_conversation(
    req: CreateConversationRequest | None = None,
    db: AsyncSession = Depends(get_db),
):
    title = (req.title if req else None) or "New conversation"
    return await create_conversation(title, db)


@router.get("/conversations", response_model=list[ConversationSummary])
async def get_conversations(db: AsyncSession = Depends(get_db)):
    return await list_conversations(db)


@router.get("/conversations/{conversation_id}", response_model=ConversationOut)
async def get_conv(conversation_id: str, db: AsyncSession = Depends(get_db)):
    conv = await get_conversation(conversation_id, db)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return conv


@router.post("/conversations/{conversation_id}/messages", response_model=ChatResponse)
async def send_message(
    conversation_id: str,
    req: ChatRequest,
    db: AsyncSession = Depends(get_db),
):
    try:
        return await post_message(
            conv_id=conversation_id,
            user_text=req.message,
            db=db,
            filter_doc_id=req.filter_doc_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/conversations/{conversation_id}", status_code=204)
async def delete_conv(conversation_id: str, db: AsyncSession = Depends(get_db)):
    deleted = await delete_conversation(conversation_id, db)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found.")
