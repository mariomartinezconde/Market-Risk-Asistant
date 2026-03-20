"""
Embedding layer.

Wraps sentence-transformers so the rest of the codebase is insulated from the
specific model being used.  Embeddings are normalised (unit vectors) so that
inner-product similarity == cosine similarity, which is what FAISS IndexFlatIP
measures.
"""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from app.logger import get_logger
from config import get_settings

logger = get_logger(__name__)

_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        settings = get_settings()
        logger.info("embeddings.loading_model", model=settings.EMBEDDING_MODEL)
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
        logger.info("embeddings.model_ready")
    return _model


def embed_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """
    Encode a list of strings and return a float32 ndarray of shape (N, D).
    Vectors are L2-normalised for cosine similarity via dot product.
    """
    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,  # ← critical for cosine similarity
    ).astype(np.float32)
    return embeddings


def embed_query(query: str) -> np.ndarray:
    """Encode a single query string. Returns shape (1, D)."""
    return embed_texts([query])
