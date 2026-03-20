"""
Query Orchestrator
==================
The single entry point for answering a user query end-to-end.

Pipeline:
  1. Retrieve chunks from the vector store (with optional metadata filters).
  2. Governance gate: if no chunks retrieved → return INSUFFICIENT answer.
  3. Detect conflicts in retrieved chunks.
  4. Build LLM prompt (system + user with injected context).
  5. Call Claude.
  6. Format the structured response.
  7. Write full audit record.

This module is intentionally framework-agnostic (no FastAPI imports) so it
can be tested without spinning up the web server.
"""
from __future__ import annotations

from app.logger import get_audit_logger, get_logger
from app.models import QueryRequest, QueryResponse
from config import get_settings
from llm.answer_formatter import format_response
from llm.claude_client import call_claude
from llm.conflict_detector import detect_conflicts
from llm.prompts import NO_CONTEXT_ANSWER, SYSTEM_PROMPT, build_user_message
from retrieval.vector_store import get_vector_store

logger = get_logger(__name__)


def answer_query(request: QueryRequest) -> QueryResponse:
    settings = get_settings()
    store = get_vector_store()
    audit = get_audit_logger(settings.AUDIT_LOG_PATH)

    # ── 1. Retrieve ────────────────────────────────────────────────────────
    top_k = request.top_k or settings.TOP_K_RETRIEVAL
    chunks = store.search(
        query=request.query,
        top_k=top_k,
        filter_doc_type=request.filter_doc_type,
        filter_doc_name=request.filter_doc_name,
    )

    # ── 2. Governance gate: no-context ────────────────────────────────────
    if not chunks:
        logger.warning("orchestrator.no_context", query=request.query[:80])
        response = format_response(
            query=request.query,
            raw_answer=NO_CONTEXT_ANSWER,
            chunks=[],
            conflict_detected=False,
            conflict_details=None,
            no_context=True,
        )
        _audit(audit, request, chunks, response, user_message=None, raw_answer=NO_CONTEXT_ANSWER)
        return response

    # ── 3. Conflict detection ─────────────────────────────────────────────
    conflict_detected, conflict_details = (
        detect_conflicts(chunks)
        if settings.CONFLICT_DETECTION_ENABLED
        else (False, None)
    )

    # ── 4. Build prompt ───────────────────────────────────────────────────
    user_message = build_user_message(
        query=request.query,
        chunks=chunks,
        conflict_hint=conflict_details if conflict_detected else None,
    )

    # ── 5. Call LLM ───────────────────────────────────────────────────────
    raw_answer = call_claude(
        system_prompt=SYSTEM_PROMPT,
        user_message=user_message,
    )

    # ── 6. Format response ────────────────────────────────────────────────
    response = format_response(
        query=request.query,
        raw_answer=raw_answer,
        chunks=chunks,
        conflict_detected=conflict_detected,
        conflict_details=conflict_details,
    )

    # ── 7. Audit ──────────────────────────────────────────────────────────
    _audit(audit, request, chunks, response, user_message, raw_answer)

    return response


def _audit(audit, request, chunks, response, user_message, raw_answer):
    from retrieval.vector_store import RetrievedChunk

    audit.log({
        "query_id": response.query_id,
        "query": request.query,
        "filter_doc_type": request.filter_doc_type.value if request.filter_doc_type else None,
        "filter_doc_name": request.filter_doc_name,
        "chunks_retrieved": [
            {
                "chunk_id": rc.chunk.chunk_id,
                "doc_name": rc.chunk.metadata.doc_name,
                "score": rc.score,
                "excerpt": rc.chunk.text[:200],
            }
            for rc in chunks
        ],
        "prompt_user_message": user_message,
        "raw_llm_answer": raw_answer,
        "confidence": response.confidence.value,
        "conflict_detected": response.conflict_detected,
        "conflict_details": response.conflict_details,
        "escalation_recommended": response.escalation_recommended,
        "escalation_reason": response.escalation_reason,
    })
