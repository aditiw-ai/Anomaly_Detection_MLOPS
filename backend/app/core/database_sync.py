"""
Synchronous database operations for Celery workers.
Celery workers need sync DB access to avoid event loop conflicts.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from app.core.config import settings

# Convert async URL to sync URL (psycopg2 uses sslmode, not ssl)
sync_database_url = (
    settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql+psycopg2://")
    .replace("?ssl=require", "?sslmode=require")
    .replace("&ssl=require", "&sslmode=require")
)

# Create sync engine for Celery workers
sync_engine = create_engine(
    sync_database_url,
    pool_size=settings.DATABASE_POOL_SIZE,
    max_overflow=settings.DATABASE_MAX_OVERFLOW,
    pool_pre_ping=True,
    echo=settings.DEBUG,
)

# Sync session factory
SyncSessionLocal = sessionmaker(
    bind=sync_engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


def get_sync_db() -> Session:
    """Get synchronous database session for Celery workers."""
    db = SyncSessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
