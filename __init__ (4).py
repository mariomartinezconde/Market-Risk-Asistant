"""
FAISS vector store.
Persists index + JSON metadata to VECTORSTORE_DIR.
Supports per-document delete and rebuild.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from app.core.config import get_settings
from app.core.logging import get_logger
from app.ingestion.chunker import Chunk

logger = get_logger(__name__)

INDEX_FILE = "index.faiss"
META_FILE = "metadata.json"


class VectorStore:
    def __init__(self) -> None:
        s = get_settings()
        self._dir = s.vectorstore_path
        self._index: faiss.Index | None = None
        self._chunks: list[Chunk] = []

    # ── Persistence ───────────────────────────────────────────────────────

    def load(self) -> bool:
        idx_path = self._dir / INDEX_FILE
        meta_path = self._dir / META_FILE
        if not idx_path.exists() or not meta_path.exists():
            return False
        try:
            self._index = faiss.read_index(str(idx_path))
            raw = json.loads(meta_path.read_text())
            self._chunks = [Chunk(**r) for r in raw]
            logger.info("vectorstore.loaded", chunks=len(self._chunks))
            return True
        except Exception as e:
            logger.error("vectorstore.load_failed", error=str(e))
            return False

    def save(self) -> None:
        if self._index is None:
            return
        faiss.write_index(self._index, str(self._dir / INDEX_FILE))
        meta = [vars(c) for c in self._chunks]
        (self._dir / META_FILE).write_text(
            json.dumps(meta, indent=2, ensure_ascii=False)
        )

    # ── Build / Add ───────────────────────────────────────────────────────

    def add_chunks(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Add new chunks. Rebuilds index from all chunks + new ones."""
        settings = get_settings()
        dim = embeddings.shape[1]

        if self._index is None:
            self._index = faiss.IndexFlatIP(dim)

        self._chunks.extend(chunks)
        self._index.add(embeddings)
        self.save()
        logger.info("vectorstore.chunks_added", added=len(chunks), total=len(self._chunks))

    def rebuild_for_doc(self, doc_id: str, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Remove all chunks for doc_id and replace with new ones."""
        self._remove_doc_chunks(doc_id)
        if not chunks:
            self.save()
            return

        settings = get_settings()
        dim = embeddings.shape[1]

        # Re-embed existing chunks and rebuild full index
        if self._chunks:
            from app.vectorstore.embedder import embed_texts
            existing_texts = [c.text for c in self._chunks]
            existing_embs = embed_texts(existing_texts)
            all_chunks = self._chunks + chunks
            all_embs = np.vstack([existing_embs, embeddings])
        else:
            all_chunks = chunks
            all_embs = embeddings

        self._chunks = all_chunks
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(all_embs.astype(np.float32))
        self.save()

    def _remove_doc_chunks(self, doc_id: str) -> None:
        self._chunks = [c for c in self._chunks if c.doc_id != doc_id]
        # Rebuild index from scratch without removed doc
        if self._chunks:
            from app.vectorstore.embedder import embed_texts
            texts = [c.text for c in self._chunks]
            embs = embed_texts(texts)
            dim = embs.shape[1]
            self._index = faiss.IndexFlatIP(dim)
            self._index.add(embs.astype(np.float32))
        else:
            self._index = None
        logger.info("vectorstore.doc_removed", doc_id=doc_id, remaining=len(self._chunks))

    def remove_doc(self, doc_id: str) -> None:
        self._remove_doc_chunks(doc_id)
        self.save()

    # ── Search ────────────────────────────────────────────────────────────

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int,
        filter_doc_id: Optional[str] = None,
        filter_doc_type: Optional[str] = None,
    ) -> list[tuple[Chunk, float]]:
        if self._index is None or not self._chunks:
            return []

        settings = get_settings()
        over_fetch = min(top_k * 5, len(self._chunks))
        scores, indices = self._index.search(query_vec, over_fetch)

        results: list[tuple[Chunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or score < settings.MIN_SIMILARITY_SCORE:
                continue
            chunk = self._chunks[idx]
            if filter_doc_id and chunk.doc_id != filter_doc_id:
                continue
            if filter_doc_type and not hasattr(chunk, 'doc_type'):
                pass  # doc_type is on the DB model, not the chunk — skip filter here
            results.append((chunk, float(score)))
            if len(results) >= top_k:
                break
        return results

    # ── Stats ─────────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._index is not None

    @property
    def total_chunks(self) -> int:
        return len(self._chunks)

    def chunks_for_doc(self, doc_id: str) -> int:
        return sum(1 for c in self._chunks if c.doc_id == doc_id)


_store: VectorStore | None = None


def get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
        _store.load()
    return _store
