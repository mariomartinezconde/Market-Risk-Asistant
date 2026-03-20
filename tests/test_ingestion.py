"""
Tests for the document ingestion pipeline.
Does not require Anthropic API key or FAISS index.
"""
from __future__ import annotations

import os
import tempfile
import pytest

from retrieval.ingestion import ingest_directory, _classify_document, _extract_version
from app.models import DocumentType


# ── Unit: _classify_document ─────────────────────────────────────────────────

def test_classify_regulation():
    assert _classify_document("crr_article_92", "capital requirements regulation article") == DocumentType.REGULATION

def test_classify_policy():
    assert _classify_document("internal_risk_policy", "this policy governs internal market risk") == DocumentType.INTERNAL_POLICY

def test_classify_procedure():
    assert _classify_document("sop_var_reporting", "procedure for reporting VaR to CRO") == DocumentType.PROCEDURE

def test_classify_unknown():
    assert _classify_document("meeting_notes_2024", "discussed lunch options") == DocumentType.UNKNOWN


# ── Unit: _extract_version ────────────────────────────────────────────────────

def test_extract_version_standard():
    assert _extract_version("Document\nVersion: 2.1\nDate: 2024-01-01") == "2.1"

def test_extract_version_missing():
    assert _extract_version("No version info here") is None

def test_extract_version_v_prefix():
    assert _extract_version("Version v3.0 applies from Q1 2024") == "3.0"


# ── Integration: ingest_directory with txt files ──────────────────────────────

def test_ingest_txt_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a simple text file
        (os.path.join(tmpdir, "regulation_CRR_test.txt"))
        with open(os.path.join(tmpdir, "regulation_CRR_test.txt"), "w") as f:
            f.write(
                "Article 92: Capital Requirements.\n"
                "Institutions must maintain a Total Capital Ratio of at least 8%.\n" * 20
            )

        chunks = list(ingest_directory(tmpdir, chunk_size=256, chunk_overlap=32))

        assert len(chunks) > 0, "Expected at least one chunk from the text file"
        assert all(c.text.strip() for c in chunks), "No empty chunks allowed"
        assert all(c.metadata.source_file for c in chunks), "Each chunk must have a source file"
        assert chunks[0].metadata.doc_type == DocumentType.REGULATION


def test_ingest_empty_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        chunks = list(ingest_directory(tmpdir))
        assert chunks == [], "Empty directory must yield zero chunks"


def test_ingest_chunk_ids_unique():
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "doc_a.txt"), "w") as f:
            f.write("Alpha content. " * 50)
        with open(os.path.join(tmpdir, "doc_b.txt"), "w") as f:
            f.write("Beta content. " * 50)

        chunks = list(ingest_directory(tmpdir, chunk_size=128, chunk_overlap=16))
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be globally unique"
