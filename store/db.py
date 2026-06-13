"""Database engine and session factory."""
from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from eval.config import get_settings
from store.models import Base


@lru_cache(maxsize=1)
def get_engine():
    settings = get_settings()
    return create_engine(settings.eval_db_url, pool_pre_ping=True)


@lru_cache(maxsize=1)
def _get_session_factory() -> sessionmaker:
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    factory = _get_session_factory()
    session: Session = factory()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    """Create all tables (idempotent). Use Alembic for migrations in production."""
    Base.metadata.create_all(get_engine())
