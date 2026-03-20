"""
Pytest fixtures shared across the test suite.

Fixtures:
  - settings        : Settings with test overrides (no real API key needed for most tests)
  - sample_chunks   : 4 pre-built DocumentChunk objects covering 2 doc types
  - mock_store      : FAISSVectorStore pre-loaded with sample_chunks (no disk I/O)
  - client          : TestClient for FastAPI route testing (vector store mocked)
"""
from __future__ import annotations

import os
import pytest

# ── Ensure dummy API key so Settings doesn't fail at import ────────────────
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-test-00000000000000000000000000000000")
os.environ.setdefault("APP_ENV", "test")


from app.models import DocumentChunk, DocumentMetadata, DocumentType


SAMPLE_CHUNKS: list[DocumentChunk] = [
    DocumentChunk(
        chunk_id="crr_0",
        text=(
            "Article 92 of the Capital Requirements Regulation (CRR) requires institutions "
            "to maintain a Total Capital Ratio of at least 8% at all times. "
            "The Tier 1 Capital Ratio minimum is set at 6%."
        ),
        metadata=DocumentMetadata(
            doc_id="crr001",
            doc_name="CRR_Article_92",
            doc_type=DocumentType.REGULATION,
            version="2.1",
            source_file="data/documents/CRR_Article_92.pdf",
            chunk_index=0,
            page_number=1,
            section_title="Article 92 – Own funds requirements",
            hierarchy="CRR > Article 92 > Own funds requirements",
        ),
    ),
    DocumentChunk(
        chunk_id="crr_1",
        text=(
            "Institutions shall calculate their capital requirements for market risk "
            "using either the Standardised Approach or, where approved by the competent "
            "authority, the Internal Model Approach as specified in Part Three Title IV."
        ),
        metadata=DocumentMetadata(
            doc_id="crr001",
            doc_name="CRR_Article_92",
            doc_type=DocumentType.REGULATION,
            version="2.1",
            source_file="data/documents/CRR_Article_92.pdf",
            chunk_index=1,
            page_number=2,
            section_title="Article 325 – Approaches for calculating own funds requirements for market risk",
            hierarchy="CRR > Article 325",
        ),
    ),
    DocumentChunk(
        chunk_id="policy_0",
        text=(
            "Internal Market Risk Policy v3.0: The institution's Total Capital Ratio "
            "internal target is set at 10%, providing a 200 basis point buffer above "
            "the regulatory minimum of 8%. This target shall be reviewed annually."
        ),
        metadata=DocumentMetadata(
            doc_id="pol001",
            doc_name="Internal_Market_Risk_Policy",
            doc_type=DocumentType.INTERNAL_POLICY,
            version="3.0",
            source_file="data/documents/Internal_Market_Risk_Policy.docx",
            chunk_index=0,
            page_number=1,
            section_title="Capital Targets",
            hierarchy="Internal_Market_Risk_Policy > Capital Targets",
        ),
    ),
    DocumentChunk(
        chunk_id="proc_0",
        text=(
            "Procedure MR-001: Market Risk Reporting. Risk officers must submit daily "
            "VaR reports to the Chief Risk Officer by 09:00 CET. Reports must include "
            "the 1-day 99% VaR and the 10-day 99% VaR for each trading book."
        ),
        metadata=DocumentMetadata(
            doc_id="proc001",
            doc_name="MR-001_Market_Risk_Reporting",
            doc_type=DocumentType.PROCEDURE,
            version="1.2",
            source_file="data/documents/MR-001.docx",
            chunk_index=0,
            page_number=1,
            section_title="Reporting Obligations",
            hierarchy="MR-001 > Reporting Obligations",
        ),
    ),
]


@pytest.fixture
def sample_chunks() -> list[DocumentChunk]:
    return SAMPLE_CHUNKS


@pytest.fixture
def mock_store(sample_chunks, monkeypatch):
    """
    Build an in-memory vector store from sample_chunks.
    This requires the embedding model to load (~80 MB) but avoids disk I/O.
    Mark tests that use this fixture with @pytest.mark.integration if slow.
    """
    from retrieval.vector_store import FAISSVectorStore, get_vector_store
    import retrieval.vector_store as vs_module

    store = FAISSVectorStore()
    store.build(sample_chunks)

    monkeypatch.setattr(vs_module, "_store", store)
    return store


@pytest.fixture
def client(mock_store):
    """FastAPI TestClient with the vector store pre-loaded."""
    from fastapi.testclient import TestClient
    from main import app
    return TestClient(app)
