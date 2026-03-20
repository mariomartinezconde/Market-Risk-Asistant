"""
Shared data models (Pydantic schemas) used across the entire application.
These are the canonical shapes for chunks, queries, and API responses.
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum


# ── Document / Knowledge Base ──────────────────────────────────────────────────

class DocumentType(str, Enum):
    REGULATION = "regulation"       # CRR, Basel III, RTS …
    INTERNAL_POLICY = "internal_policy"
    PROCEDURE = "procedure"
    GUIDANCE = "guidance"
    UNKNOWN = "unknown"


class DocumentMetadata(BaseModel):
    """Metadata attached to every chunk stored in the vector index."""
    doc_id: str
    doc_name: str
    doc_type: DocumentType = DocumentType.UNKNOWN
    version: Optional[str] = None
    effective_date: Optional[str] = None
    owner: Optional[str] = None
    source_file: str
    chunk_index: int
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    hierarchy: Optional[str] = None   # e.g. "CRR > Article 325 > Para 2"
    ingested_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class DocumentChunk(BaseModel):
    """A text chunk with its metadata, stored in the vector DB."""
    chunk_id: str
    text: str
    metadata: DocumentMetadata
    embedding: Optional[list[float]] = None   # excluded from serialised responses


# ── Retrieval ─────────────────────────────────────────────────────────────────

class RetrievedChunk(BaseModel):
    """A chunk returned by the retrieval layer, with its similarity score."""
    chunk: DocumentChunk
    score: float = Field(ge=0.0, le=1.0)


class RetrievalResult(BaseModel):
    chunks: list[RetrievedChunk]
    query: str
    total_found: int


# ── API request / response ────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=5,
        max_length=2000,
        description="Natural language question about market risk regulations.",
        examples=["What are the capital requirements under CRR Article 325?"]
    )
    filter_doc_type: Optional[DocumentType] = Field(
        default=None,
        description="Restrict retrieval to a specific document type."
    )
    filter_doc_name: Optional[str] = Field(
        default=None,
        description="Restrict retrieval to a specific document by name."
    )
    top_k: Optional[int] = Field(default=None, ge=1, le=20)


class SourceCitation(BaseModel):
    doc_name: str
    doc_type: str
    version: Optional[str]
    section_title: Optional[str]
    chunk_excerpt: str = Field(description="First 300 chars of the supporting chunk.")
    similarity_score: float
    hierarchy: Optional[str]


class ConfidenceLevel(str, Enum):
    HIGH = "high"           # ≥2 corroborating chunks, score ≥ 0.70
    MEDIUM = "medium"       # 1 chunk or score 0.50–0.70
    LOW = "low"             # score < 0.50 or only marginal matches
    INSUFFICIENT = "insufficient"   # below MIN_SIMILARITY threshold entirely


class QueryResponse(BaseModel):
    """
    Canonical API response. Designed to be audit-friendly and traceable.
    Every field that influences the answer is logged.
    """
    answer: str
    sources: list[SourceCitation]
    confidence: ConfidenceLevel
    conflict_detected: bool = Field(
        description="True if retrieved chunks contain contradictory statements."
    )
    conflict_details: Optional[str] = Field(
        default=None,
        description="Human-readable description of the detected conflict."
    )
    escalation_recommended: bool = Field(
        description="True when confidence is LOW/INSUFFICIENT or a conflict is detected."
    )
    escalation_reason: Optional[str] = None
    query_id: str = Field(description="UUID for this query – used in audit logs.")
    processed_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class HealthResponse(BaseModel):
    status: str
    version: str
    index_loaded: bool
    total_chunks_indexed: int
    model: str


class IngestRequest(BaseModel):
    """Body for the /ingest endpoint (used during admin setup)."""
    directory: Optional[str] = None   # defaults to DATA_DIR from settings


class IngestResponse(BaseModel):
    status: str
    documents_processed: int
    chunks_created: int
    errors: list[str]
