"""
Async SQLite database engine via SQLAlchemy + aiosqlite.
Tables are created on startup.
"""
from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

from app.core.config import get_settings

_engine = None
_session_factory = None


def get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        db_url = f"sqlite+aiosqlite:///{settings.db_path}"
        _engine = create_async_engine(db_url, echo=settings.DEBUG, future=True)
    return _engine


def get_session_factory() -> async_sessionmaker:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_engine(), expire_on_commit=False, class_=AsyncSession
        )
    return _session_factory


async def get_db() -> AsyncSession:
    async with get_session_factory()() as session:
        yield session


async def create_tables() -> None:
    from app.models.db_models import Base
    async with get_engine().begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


class Base(DeclarativeBase):
    pass
