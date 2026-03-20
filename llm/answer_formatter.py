"""
Answer Formatter
================
Transforms the raw LLM text + retrieval metadata into the structured
QueryResponse that the API returns.

Responsibilities:
  1. Compute confidence level from retrieval scores and chunk count.
  2. Build SourceCitation objects from RetrievedChunks.
  3. Determine escalation flag.
  4. Wrap everything in QueryResponse.
"""
from __future__ import annotations

import uuid

from app.models import (
    ConfidenceLevel,
    QueryResponse,
    RetrievedChunk,
    SourceCitation,
)
from config import get_settings


def _compute_confidence(chunks: list[RetrievedChunk]) -> ConfidenceLevel:
    """
    Scoring rules:
      HIGH   : ≥ 2 chunks, top score ≥ 0.70
      MEDIUM : 1 chunk, or top score in [0.50, 0.70)
      LOW    : top score in [MIN_SCORE, 0.50)
      INSUFFICIENT : no chunks (should have been caught earlier)
    """
    if not chunks:
        return ConfidenceLevel.INSUFFICIENT

    settings = get_settings()
    top_score = max(rc.score for rc in chunks)
    n = len(chunks)

    if n >= 2 and top_score >= 0.70:
        return ConfidenceLevel.HIGH
    if n >= 1 and top_score >= 0.50:
        return ConfidenceLevel.MEDIUM
    if top_score >= settings.MIN_SIMILARITY_SCORE:
        return ConfidenceLevel.LOW
    return ConfidenceLevel.INSUFFICIENT


def _build_citations(chunks: list[RetrievedChunk]) -> list[SourceCitation]:
    seen: set[str] = set()
    citations: list[SourceCitation] = []
    for rc in chunks:
        m = rc.chunk.metadata
        key = f"{m.doc_name}|{m.chunk_index}"
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            SourceCitation(
                doc_name=m.doc_name,
                doc_type=m.doc_type.value,
                version=m.version,
                section_title=m.section_title,
                chunk_excerpt=rc.chunk.text[:300].strip(),
                similarity_score=round(rc.score, 4),
                hierarchy=m.hierarchy,
            )
        )
    return citations


def format_response(
    query: str,
    raw_answer: str,
    chunks: list[RetrievedChunk],
    conflict_detected: bool,
    conflict_details: str | None,
    no_context: bool = False,
) -> QueryResponse:
    """
    Assemble the final QueryResponse.

    Parameters
    ----------
    query           : original user query
    raw_answer      : text returned by Claude
    chunks          : retrieved chunks used as context
    conflict_detected : output of conflict_detector
    conflict_details  : human-readable conflict description
    no_context      : True when no chunks were found (governance gate)
    """
    confidence = _compute_confidence(chunks) if not no_context else ConfidenceLevel.INSUFFICIENT

    escalation = (
        confidence in (ConfidenceLevel.LOW, ConfidenceLevel.INSUFFICIENT)
        or conflict_detected
        or no_context
    )

    escalation_reason: str | None = None
    if no_context:
        escalation_reason = "No relevant documents found in the knowledge base."
    elif confidence == ConfidenceLevel.LOW:
        escalation_reason = "Retrieved evidence has low similarity to the query."
    elif confidence == ConfidenceLevel.INSUFFICIENT:
        escalation_reason = "Insufficient evidence to answer reliably."
    elif conflict_detected:
        escalation_reason = "Conflicting information detected between source documents."

    return QueryResponse(
        answer=raw_answer,
        sources=_build_citations(chunks),
        confidence=confidence,
        conflict_detected=conflict_detected,
        conflict_details=conflict_details,
        escalation_recommended=escalation,
        escalation_reason=escalation_reason,
        query_id=str(uuid.uuid4()),
    )
