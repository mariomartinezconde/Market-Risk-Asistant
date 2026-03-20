"""
Central configuration for the Market Risk AI Assistant.
All environment variables are defined here.
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    # ── API ────────────────────────────────────────────────────────────────
    APP_NAME: str = "Market Risk AI Assistant"
    APP_VERSION: str = "1.0.0"
    APP_ENV: str = Field(default="development", alias="APP_ENV")
    DEBUG: bool = Field(default=False)
    API_PORT: int = Field(default=8000)
    API_HOST: str = Field(default="0.0.0.0")

    # ── Anthropic / Claude ─────────────────────────────────────────────────
    ANTHROPIC_API_KEY: str = Field(..., description="Anthropic API key (required)")
    CLAUDE_MODEL: str = Field(default="claude-opus-4-5")
    CLAUDE_MAX_TOKENS: int = Field(default=2048)
    CLAUDE_TEMPERATURE: float = Field(default=0.0)   # determinism is critical in regulated env

    # ── Embeddings ─────────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace sentence-transformer model for embeddings"
    )
    EMBEDDING_DIMENSION: int = Field(default=384)

    # ── Vector Store ───────────────────────────────────────────────────────
    FAISS_INDEX_PATH: str = Field(default="data/faiss_index")
    FAISS_METADATA_PATH: str = Field(default="data/faiss_metadata.json")
    TOP_K_RETRIEVAL: int = Field(default=8)
    MIN_SIMILARITY_SCORE: float = Field(default=0.35)

    # ── Document Ingestion ─────────────────────────────────────────────────
    DATA_DIR: str = Field(default="data/documents")
    CHUNK_SIZE: int = Field(default=512)
    CHUNK_OVERLAP: int = Field(default=64)

    # ── Governance ─────────────────────────────────────────────────────────
    AUDIT_LOG_PATH: str = Field(default="logs/audit.jsonl")
    REQUIRE_CITATIONS: bool = Field(default=True)
    CONFLICT_DETECTION_ENABLED: bool = Field(default=True)
    MIN_CONTEXT_CHUNKS_TO_ANSWER: int = Field(default=1)

    # ── Security ───────────────────────────────────────────────────────────
    API_KEY_HEADER: str = Field(default="X-API-Key")
    API_KEY: Optional[str] = Field(default=None, description="Optional API key to protect endpoints")
    ALLOWED_ORIGINS: list[str] = Field(default=["*"])

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        populate_by_name = True


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
