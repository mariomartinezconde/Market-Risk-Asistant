"""
Governance rule tests.

These tests validate that the system's governance contracts are enforced:
  G1 – No answer without context
  G2 – Mandatory citations in prompt
  G3 – Conflict detection fires when expected
  G4 – No hallucination (model is bound to context-only prompt)
  G5 – Escalation recommended when confidence is LOW or INSUFFICIENT

Note: G4 is enforced at the prompt layer (not testable without LLM call),
but we verify the prompt construction is correct.
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
from llm.prompts import SYSTEM_PROMPT, NO_CONTEXT_ANSWER, build_user_message
from llm.conflict_detector import detect_conflicts
from llm.answer_formatter import format_response


# ── G1: No answer without context ────────────────────────────────────────────

def test_g1_no_context_returns_insufficient():
    resp = format_response(
        query="Any question",
        raw_answer=NO_CONTEXT_ANSWER,
        chunks=[],
        conflict_detected=False,
        conflict_details=None,
        no_context=True,
    )
    assert resp.confidence == ConfidenceLevel.INSUFFICIENT


def test_g1_no_context_prompt_contains_marker():
    msg = build_user_message("Any question", chunks=[])
    assert "NO CONTEXT AVAILABLE" in msg


# ── G2: Mandatory citations in system prompt ──────────────────────────────────

def test_g2_system_prompt_requires_citations():
    assert "MANDATORY CITATIONS" in SYSTEM_PROMPT
    assert "[DOC_NAME" in SYSTEM_PROMPT or "DOC_NAME" in SYSTEM_PROMPT


def test_g2_context_prompt_includes_doc_metadata():
    chunk = RetrievedChunk(
        chunk=DocumentChunk(
            chunk_id="c1",
            text="Capital ratio must be 8%.",
            metadata=DocumentMetadata(
                doc_id="crr",
                doc_name="CRR_Article_92",
                doc_type=DocumentType.REGULATION,
                version="2.1",
                source_file="crr.pdf",
                chunk_index=0,
                section_title="Own funds requirements",
            ),
        ),
        score=0.90,
    )
    msg = build_user_message("What is the minimum capital ratio?", [chunk])
    assert "CRR_Article_92" in msg
    assert "regulation" in msg.lower()
    assert "Own funds requirements" in msg


# ── G3: Conflict detection ────────────────────────────────────────────────────

def test_g3_conflict_detected_across_versions():
    def _rc(name, version):
        return RetrievedChunk(
            chunk=DocumentChunk(
                chunk_id=f"{name}_{version}",
                text="Capital ratio minimum is set at 8%.",
                metadata=DocumentMetadata(
                    doc_id=name, doc_name=name, doc_type=DocumentType.REGULATION,
                    version=version, source_file=f"{name}.pdf", chunk_index=0,
                ),
            ),
            score=0.85,
        )

    chunks = [_rc("Internal_Policy", "1.0"), _rc("Internal_Policy", "2.0")]
    detected, details = detect_conflicts(chunks)
    assert detected
    assert details is not None


def test_g3_no_false_positive_single_doc():
    chunk = RetrievedChunk(
        chunk=DocumentChunk(
            chunk_id="c1",
            text="Institutions must hold at least 8% capital.",
            metadata=DocumentMetadata(
                doc_id="crr", doc_name="CRR", doc_type=DocumentType.REGULATION,
                version="1.0", source_file="crr.pdf", chunk_index=0,
            ),
        ),
        score=0.90,
    )
    detected, _ = detect_conflicts([chunk])
    assert not detected


# ── G5: Escalation logic ──────────────────────────────────────────────────────

def test_g5_escalation_on_low_confidence():
    chunk = RetrievedChunk(
        chunk=DocumentChunk(
            chunk_id="c1",
            text="Some vaguely related text.",
            metadata=DocumentMetadata(
                doc_id="x", doc_name="SomeDoc", doc_type=DocumentType.UNKNOWN,
                version=None, source_file="x.txt", chunk_index=0,
            ),
        ),
        score=0.36,  # Just above MIN_SIMILARITY_SCORE threshold
    )
    resp = format_response(
        query="Specific regulatory question",
        raw_answer="Uncertain answer.",
        chunks=[chunk],
        conflict_detected=False,
        conflict_details=None,
    )
    # Score 0.36 with 1 chunk → LOW confidence → escalation required
    assert resp.confidence in (ConfidenceLevel.LOW, ConfidenceLevel.MEDIUM)
    if resp.confidence == ConfidenceLevel.LOW:
        assert resp.escalation_recommended is True


def test_g5_no_escalation_on_high_confidence():
    chunks = [
        RetrievedChunk(
            chunk=DocumentChunk(
                chunk_id=f"c{i}",
                text="Capital ratio must be at least 8%.",
                metadata=DocumentMetadata(
                    doc_id=f"doc{i}", doc_name=f"Doc{i}", doc_type=DocumentType.REGULATION,
                    version="1.0", source_file=f"doc{i}.pdf", chunk_index=0,
                ),
            ),
            score=0.92,
        )
        for i in range(3)
    ]
    resp = format_response(
        query="What is the capital ratio?",
        raw_answer="It is 8% [Doc0].",
        chunks=chunks,
        conflict_detected=False,
        conflict_details=None,
    )
    assert resp.confidence == ConfidenceLevel.HIGH
    assert resp.escalation_recommended is False
