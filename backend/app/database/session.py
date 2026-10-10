"""Engine/session management.

Supports PostgreSQL (production, provided by docker-compose) and SQLite
(local development fallback) through the same ``DATABASE_URL`` setting.
"""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings

_connect_args: dict = {}
_engine_kwargs: dict = {"echo": settings.db_echo, "future": True}

if settings.is_sqlite:
    # SQLite + threads: FastAPI runs sync sessions in a worker threadpool.
    _connect_args["check_same_thread"] = False
else:
    _engine_kwargs.update(pool_pre_ping=True, pool_size=10, max_overflow=20)

engine: Engine = create_engine(
    settings.database_url,
    connect_args=_connect_args,
    **_engine_kwargs,
)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
    class_=Session,
)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a scoped session."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def create_all() -> None:
    """Create every table.

    Used by tests and first-run bootstrap. Alembic handles real migrations.
    """
    from app.database.base import Base  # noqa: F401  (import registers metadata)
    import app.models  # noqa: F401

    Base.metadata.create_all(bind=engine)


def drop_all() -> None:
    from app.database.base import Base
    import app.models  # noqa: F401

    Base.metadata.drop_all(bind=engine)
