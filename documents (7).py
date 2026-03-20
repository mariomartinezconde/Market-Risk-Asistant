"""
RAG pipeline: retrieve → detect conflicts → build prompt → call LLM → format.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Optional

from app.core.config import get_settings
from app.core.logging import get_logger, get_audit_logger
from app.ingestion.chunker import Chunk
from app.llm.base import get_llm_provider, LLMNotConfiguredError
from app.rag.prompts import SYSTEM_PROMPT, NO_CONTEXT_RESPONSE, build_prompt
from app.schemas.chat import SourceCitation
from app.vectorstore.embedder import embed_query
from app.vectorstore.store import get_store

logger = get_logger(__name__)


class RAGResult:
    def __init__(
        self,
        answer: str,
        sources: list[SourceCitation],
        confidence: str,
        conflict_detected: bool,
        escalation_recommended: bool,
    ):
        self.answer = answer
        self.sources = sources
        self.confidence = confidence
        self.conflict_detected = conflict_detected
        self.escalation_recommended = escalation_recommended


# ── Document metadata cache (filled by service layer) ────────────────────────
# Maps doc_id → {name, type, version}
_doc_meta_cache: dict[str, dict] = {}


def update_doc_meta_cache(doc_id: str, name: str, doc_type: str, version: Optional[str]) -> None:
    _doc_meta_cache[doc_id] = {"name": name, "type": doc_type, "version": version}


def _get_doc_name(doc_id: str) -> str:
    return _doc_meta_cache.get(doc_id, {}).get("name", doc_id)


# ── Conflict detection ────────────────────────────────────────────────────────

def _detect_conflicts(chunks: list[tuple[Chunk, float]]) -> tuple[bool, Optional[str]]:
    if len(chunks) < 2:
        return False, None

    version_map: dict[str, set[str]] = defaultdict(set)
    for chunk, _ in chunks:
        meta = _doc_meta_cache.get(chunk.doc_id, {})
        if meta.get("version"):
            version_map[chunk.doc_id].add(meta["version"])

    threshold_map: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    pattern = re.compile(
        r"(minimum|maximum|threshold|limit|ratio)\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*(%|percent|bps)",
        re.I,
    )
    for chunk, _ in chunks:
        for m in pattern.finditer(chunk.text):
            keyword = m.group(1).lower()
            value = f"{m.group(2)}{m.group(3)}"
            threshold_map[keyword][value].add(_get_doc_name(chunk.doc_id))

    conflicts = []
    for kw, val_docs in threshold_map.items():
        if len(val_docs) > 1:
            parts = [f"{v} (in {', '.join(sorted(docs))})" for v, docs in val_docs.items()]
            conflicts.append(f"Conflicting '{kw}': {' vs '.join(parts)}")

    if conflicts:
        return True, "; ".join(conflicts)
    return False, None


# ── Confidence scoring ────────────────────────────────────────────────────────

def _score_confidence(chunks: list[tuple[Chunk, float]]) -> str:
    if not chunks:
        return "insufficient"
    top = max(s for _, s in chunks)
    n = len(chunks)
    if n >= 2 and top >= 0.70:
        return "high"
    if top >= 0.50:
        return "medium"
    if top >= get_settings().MIN_SIMILARITY_SCORE:
        return "low"
    return "insufficient"


# ── Main pipeline ─────────────────────────────────────────────────────────────

async def run_rag(
    query: str,
    conversation_id: str,
    filter_doc_id: Optional[str] = None,
) -> RAGResult:
    settings = get_settings()
    store = get_store()
    audit = get_audit_logger()

    # 1. Embed query
    query_vec = embed_query(query)

    # 2. Retrieve
    hits = store.search(
        query_vec=query_vec,
        top_k=settings.TOP_K_RETRIEVAL,
        filter_doc_id=filter_doc_id,
    )

    confidence = _score_confidence(hits)

    # 3. Governance gate
    if not hits:
        logger.warning("rag.no_context", query=query[:80])
        result = RAGResult(
            answer=NO_CONTEXT_RESPONSE,
            sources=[],
            confidence="insufficient",
            conflict_detected=False,
            escalation_recommended=True,
        )
        audit.log({"event": "rag_query", "conversation_id": conversation_id,
                   "query": query, "hits": 0, "confidence": "insufficient"})
        return result

    # 4. Conflict detection
    conflict_detected, conflict_hint = _detect_conflicts(hits)

    # 5. Build sources
    sources = []
    seen = set()
    for chunk, score in hits:
        key = f"{chunk.doc_id}_{chunk.chunk_index}"
        if key in seen:
            continue
        seen.add(key)
        meta = _doc_meta_cache.get(chunk.doc_id, {})
        sources.append(SourceCitation(
            doc_id=chunk.doc_id,
            doc_name=meta.get("name", chunk.doc_id),
            doc_type=meta.get("type", "unknown"),
            version=meta.get("version"),
            section_title=chunk.section_title,
            excerpt=chunk.text[:300].strip(),
            similarity_score=round(score, 4),
        ))

    # 6. Build prompt
    user_msg = build_prompt(query, hits, conflict_hint if conflict_detected else None)

    # 7. Call LLM
    try:
        llm = get_llm_provider()
        answer = llm.complete(SYSTEM_PROMPT, user_msg)
    except LLMNotConfiguredError as e:
        answer = f"⚠ LLM not configured: {e}"
        confidence = "insufficient"
    except Exception as e:
        logger.error("rag.llm_error", error=str(e))
        answer = "An error occurred while generating the response. Please try again."

    escalation = confidence in ("low", "insufficient") or conflict_detected

    # 8. Audit
    audit.log({
        "event": "rag_query",
        "conversation_id": conversation_id,
        "query": query,
        "hits": len(hits),
        "confidence": confidence,
        "conflict_detected": conflict_detected,
        "escalation": escalation,
    })

    return RAGResult(
        answer=answer,
        sources=sources,
        confidence=confidence,
        conflict_detected=conflict_detected,
        escalation_recommended=escalation,
    )
