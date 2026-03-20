"""
API route integration tests.

The LLM call (call_claude) is monkeypatched so these tests do not require
a real Anthropic API key or make any HTTP requests to Anthropic.
"""
from __future__ import annotations

import pytest
from unittest.mock import patch

MOCK_ANSWER = (
    "Institutions must maintain a Total Capital Ratio of at least 8% [CRR_Article_92, Article 92(1)]. "
    "The Tier 1 Capital Ratio minimum is 6% [CRR_Article_92, Article 92(1)(b)]."
)


@pytest.fixture
def client_with_llm_mock(mock_store):
    """TestClient with the LLM call mocked out."""
    with patch("llm.claude_client.call_claude", return_value=MOCK_ANSWER):
        from fastapi.testclient import TestClient
        from main import app
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


# ── /health ───────────────────────────────────────────────────────────────────

def test_health_ok(client_with_llm_mock):
    resp = client_with_llm_mock.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["index_loaded"] is True
    assert data["total_chunks_indexed"] > 0


# ── /query ────────────────────────────────────────────────────────────────────

def test_query_returns_structured_response(client_with_llm_mock):
    resp = client_with_llm_mock.post(
        "/query",
        json={"query": "What is the minimum Total Capital Ratio under CRR?"},
    )
    assert resp.status_code == 200
    data = resp.json()

    # All required fields must be present
    assert "answer" in data
    assert "sources" in data
    assert "confidence" in data
    assert "conflict_detected" in data
    assert "escalation_recommended" in data
    assert "query_id" in data
    assert "processed_at" in data

    assert isinstance(data["sources"], list)
    assert data["answer"] == MOCK_ANSWER


def test_query_too_short(client_with_llm_mock):
    resp = client_with_llm_mock.post("/query", json={"query": "hi"})
    assert resp.status_code == 422  # Pydantic validation error (min_length=5)


def test_query_with_doc_type_filter(client_with_llm_mock):
    resp = client_with_llm_mock.post(
        "/query",
        json={
            "query": "What are the capital requirements for market risk?",
            "filter_doc_type": "regulation",
        },
    )
    assert resp.status_code == 200


# ── /documents ────────────────────────────────────────────────────────────────

def test_list_documents(client_with_llm_mock):
    resp = client_with_llm_mock.get("/documents")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_documents" in data
    assert "documents" in data
    assert data["total_documents"] > 0


# ── Governance: escalation on no-context ──────────────────────────────────────

def test_escalation_on_no_context(mock_store):
    """
    When the query matches nothing in the vector store, the response must:
      - have confidence=INSUFFICIENT
      - have escalation_recommended=True
    """
    from unittest.mock import patch as mp
    from app.models import QueryRequest
    from app.orchestrator import answer_query

    # Patch the store to return zero results
    with mp.object(mock_store, "search", return_value=[]):
        import retrieval.vector_store as vs_module
        vs_module._store = mock_store

        request = QueryRequest(query="xkcd quantum foam topology in banking regulation")
        response = answer_query(request)

        assert response.confidence.value == "insufficient"
        assert response.escalation_recommended is True
        assert response.sources == []
