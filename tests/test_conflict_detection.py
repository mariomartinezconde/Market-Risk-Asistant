"""
Tests for the conflict detection module.
No external dependencies required.
"""
from __future__ import annotations

import pytest
from app.models import DocumentChunk, DocumentMetadata, DocumentType, RetrievedChunk
from llm.conflict_detector import detect_conflicts


def _make_chunk(doc_name: str, text: str, version: str = "1.0", doc_type=DocumentType.REGULATION) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=DocumentChunk(
            chunk_id=f"{doc_name}_0",
            text=text,
            metadata=DocumentMetadata(
                doc_id=doc_name,
                doc_name=doc_name,
                doc_type=doc_type,
                version=version,
                source_file=f"data/{doc_name}.pdf",
                chunk_index=0,
            ),
        ),
        score=0.85,
    )


def test_no_conflict_single_chunk():
    chunks = [_make_chunk("CRR", "Total Capital Ratio must be at least 8%.")]
    detected, detail = detect_conflicts(chunks)
    assert not detected
    assert detail is None


def test_no_conflict_consistent_values():
    chunks = [
        _make_chunk("CRR", "minimum of 8% Total Capital Ratio is required"),
        _make_chunk("BIS", "minimum of 8% ratio is the international standard"),
    ]
    detected, detail = detect_conflicts(chunks)
    # Both say 8% → no conflict
    assert not detected


def test_version_conflict_same_doc():
    chunk_v1 = _make_chunk("Internal_Policy", "Capital buffer target is 2%.", version="1.0")
    chunk_v2 = _make_chunk("Internal_Policy", "Capital buffer target is 3%.", version="2.0")
    detected, detail = detect_conflicts([chunk_v1, chunk_v2])
    assert detected
    assert "Internal_Policy" in detail
    assert "1.0" in detail and "2.0" in detail


def test_empty_chunks():
    detected, detail = detect_conflicts([])
    assert not detected
    assert detail is None


def test_conflict_returns_string_description():
    chunk_v1 = _make_chunk("PolicyA", "Liquidity ratio minimum is 100%.", version="1.0")
    chunk_v2 = _make_chunk("PolicyA", "Liquidity ratio minimum is 110%.", version="2.0")
    detected, detail = detect_conflicts([chunk_v1, chunk_v2])
    if detected:
        assert isinstance(detail, str)
        assert len(detail) > 10
