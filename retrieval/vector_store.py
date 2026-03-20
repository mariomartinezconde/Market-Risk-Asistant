"""
FAISS Vector Store
==================
Manages the FAISS index and the parallel metadata store.

Design decisions (documented):
- IndexFlatIP (exact inner-product search) is used instead of approximate
  methods (HNSW / IVF) because the corpus size (< 100k chunks) doesn't
  justify the recall trade-off in a regulated environment.
- Metadata is stored as a JSON sidecar file (not inside FAISS) because FAISS
  doesn't natively store arbitrary metadata.
- Filtering is applied *post-retrieval* on the top-k*5 candidates.  Pre-filter
  via a separate per-type index is a future optimisation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from app.logger import get_logger
from app.models import DocumentChunk, DocumentMetadata, DocumentType, RetrievedChunk
from config import get_settings
from retrieval.embeddings import embed_query, embed_texts

logger = get_logger(__name__)


class FAISSVectorStore:
    """Thread-safe (read) FAISS vector store with metadata filtering."""

    def __init__(self) -> None:
        settings = get_settings()
        self.index_path = Path(settings.FAISS_INDEX_PATH)
        self.metadata_path = Path(settings.FAISS_METADATA_PATH)
        self.top_k = settings.TOP_K_RETRIEVAL
        self.min_score = settings.MIN_SIMILARITY_SCORE

        self._index: faiss.Index | None = None
        self._chunks: list[DocumentChunk] = []   # positional – idx == FAISS row

    # ── Build / persist ──────────────────────────────────────────────────────

    def build(self, chunks: list[DocumentChunk]) -> None:
        """Embed all chunks and build a new FAISS index from scratch."""
        logger.info("vector_store.build_start", n_chunks=len(chunks))

        texts = [c.text for c in chunks]
        embeddings = embed_texts(texts)

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)
        self._chunks = chunks

        self._save()
        logger.info("vector_store.build_done", n_chunks=len(chunks), dim=dim)

    def _save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.index_path))

        metadata_path = Path(str(self.metadata_path))
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        serialised = [c.model_dump(exclude={"embedding"}) for c in self._chunks]
        metadata_path.write_text(json.dumps(serialised, indent=2, ensure_ascii=False))
        logger.info("vector_store.saved", index=str(self.index_path))

    # ── Load ─────────────────────────────────────────────────────────────────

    def load(self) -> bool:
        """Load index and metadata from disk. Returns True on success."""
        if not self.index_path.exists() or not Path(self.metadata_path).exists():
            logger.warning("vector_store.not_found_on_disk")
            return False
        try:
            self._index = faiss.read_index(str(self.index_path))
            raw = json.loads(Path(self.metadata_path).read_text())
            self._chunks = [
                DocumentChunk(
                    chunk_id=r["chunk_id"],
                    text=r["text"],
                    metadata=DocumentMetadata(**r["metadata"]),
                )
                for r in raw
            ]
            logger.info("vector_store.loaded", n_chunks=len(self._chunks))
            return True
        except Exception as exc:
            logger.error("vector_store.load_failed", error=str(exc))
            return False

    # ── Query ────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_doc_type: Optional[DocumentType] = None,
        filter_doc_name: Optional[str] = None,
    ) -> list[RetrievedChunk]:
        """
        Semantic search with optional metadata filtering.

        Strategy:
          1. Embed the query.
          2. Retrieve top_k * 5 candidates from FAISS (to allow post-filtering).
          3. Apply metadata filters.
          4. Score-threshold: drop chunks below MIN_SIMILARITY_SCORE.
          5. Return top_k results.
        """
        if self._index is None:
            raise RuntimeError("Vector store is not loaded. Call load() or build() first.")

        top_k = top_k or self.top_k
        over_fetch = min(top_k * 5, len(self._chunks))

        query_vec = embed_query(query)   # shape (1, D)
        scores, indices = self._index.search(query_vec, over_fetch)
        scores = scores[0]    # flatten
        indices = indices[0]

        results: list[RetrievedChunk] = []
        for score, idx in zip(scores, indices):
            if idx < 0:           # FAISS returns -1 for padding
                continue
            if score < self.min_score:
                continue

            chunk = self._chunks[idx]

            # Metadata filters
            if filter_doc_type and chunk.metadata.doc_type != filter_doc_type:
                continue
            if filter_doc_name and filter_doc_name.lower() not in chunk.metadata.doc_name.lower():
                continue

            results.append(RetrievedChunk(chunk=chunk, score=float(score)))
            if len(results) >= top_k:
                break

        logger.info(
            "vector_store.search",
            query=query[:80],
            results=len(results),
            filter_doc_type=filter_doc_type,
            filter_doc_name=filter_doc_name,
        )
        return results

    # ── Introspection ────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._index is not None

    @property
    def total_chunks(self) -> int:
        return len(self._chunks)


# ── Singleton ────────────────────────────────────────────────────────────────

_store: FAISSVectorStore | None = None


def get_vector_store() -> FAISSVectorStore:
    global _store
    if _store is None:
        _store = FAISSVectorStore()
    return _store
