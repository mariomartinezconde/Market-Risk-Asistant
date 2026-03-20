"""
Tests for the answer formatter and confidence scoring.
No external dependencies required.
"""
from __future__ import annotations

import pytest
from app.models import (
    ConfidenceLevel,
    DocumentChunk,
    DocumentMetadata,
    DocumentType,
    RetrievedChunk,
)
from llm.answer_formatter import _compute_confidence, _build_citations, format_response


def _make_rc(score: float, doc_name: str = "CRR") -> RetrievedChunk:
    return RetrievedChunk(
        chunk=DocumentChunk(
            chunk_id=f"{doc_name}_{score}",
            text=f"Sample text from {doc_name} with score {score}.",
            metadata=DocumentMetadata(
                doc_id=doc_name,
                doc_name=doc_name,
                doc_type=DocumentType.REGULATION,
                version="1.0",
                source_file=f"data/{doc_name}.pdf",
                chunk_index=0,
                section_title="Capital Requirements",
            ),
        ),
        score=score,
    )


# ── _compute_confidence ───────────────────────────────────────────────────────

def test_confidence_high_two_chunks():
    chunks = [_make_rc(0.85), _make_rc(0.80, "BIS")]
    assert _compute_confidence(chunks) == ConfidenceLevel.HIGH


def test_confidence_medium_single_chunk():
    chunks = [_make_rc(0.65)]
    assert _compute_confidence(chunks) == ConfidenceLevel.MEDIUM


def test_confidence_low_score():
    chunks = [_make_rc(0.38)]
    assert _compute_confidence(chunks) == ConfidenceLevel.LOW


def test_confidence_insufficient_empty():
    assert _compute_confidence([]) == ConfidenceLevel.INSUFFICIENT


# ── _build_citations ──────────────────────────────────────────────────────────

def test_citations_deduplication():
    # Same chunk twice should produce one citation
    rc = _make_rc(0.90)
    citations = _build_citations([rc, rc])
    assert len(citations) == 1


def test_citations_contain_excerpt():
    rc = _make_rc(0.90, "CRR")
    citations = _build_citations([rc])
    assert citations[0].chunk_excerpt != ""
    assert citations[0].doc_name == "CRR"


# ── format_response ───────────────────────────────────────────────────────────

def test_format_response_no_context():
    resp = format_response(
        query="What is VaR?",
        raw_answer="I don't know.",
        chunks=[],
        conflict_detected=False,
        conflict_details=None,
        no_context=True,
    )
    assert resp.confidence == ConfidenceLevel.INSUFFICIENT
    assert resp.escalation_recommended is True
    assert resp.sources == []


def test_format_response_with_conflict():
    chunks = [_make_rc(0.85), _make_rc(0.80, "Internal_Policy")]
    resp = format_response(
        query="What is the minimum capital ratio?",
        raw_answer="There is a conflict between documents.",
        chunks=chunks,
        conflict_detected=True,
        conflict_details="CRR says 8%, policy says 10%.",
    )
    assert resp.conflict_detected is True
    assert resp.escalation_recommended is True
    assert resp.conflict_details == "CRR says 8%, policy says 10%."


def test_format_response_query_id_is_uuid():
    import uuid
    chunks = [_make_rc(0.90)]
    resp = format_response(
        query="test", raw_answer="answer", chunks=chunks,
        conflict_detected=False, conflict_details=None,
    )
    # Should not raise
    uuid.UUID(resp.query_id)
