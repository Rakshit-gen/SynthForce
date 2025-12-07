"""
Database connection and session management.

Provides async SQLAlchemy engine, session factory, and connection utilities.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from app.config import get_settings
from app.models.orm import Base

logger = logging.getLogger(__name__)

# Global engine instance
_engine: Optional[AsyncEngine] = None
_session_factory: Optional[async_sessionmaker[AsyncSession]] = None


def get_engine() -> AsyncEngine:
    """
    Get or create the async database engine.
    
    Uses connection pooling for production, NullPool for testing.
    """
    global _engine
    
    if _engine is None:
        settings = get_settings()
        
        # Engine configuration based on environment
        engine_kwargs = {
            "echo": settings.database.echo or settings.debug,
            "future": True,
        }
        
        if settings.is_production:
            # Use connection pooling in production
            engine_kwargs.update({
                "pool_size": settings.database.pool_min,
                "max_overflow": settings.database.pool_max - settings.database.pool_min,
                "pool_timeout": settings.database.pool_timeout,
                "pool_pre_ping": True,
                "pool_recycle": 3600,
            })
        else:
            # Use NullPool for development/testing to avoid connection issues
            engine_kwargs["poolclass"] = NullPool
        
        _engine = create_async_engine(
            settings.database.url,
            **engine_kwargs
        )
        
        logger.info(
            f"Database engine created",
            extra={
                "database_url": settings.database.url.split("@")[-1],  # Hide credentials
                "environment": settings.environment,
            }
        )
    
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """
    Get or create the async session factory.
    """
    global _session_factory
    
    if _session_factory is None:
        engine = get_engine()
        _session_factory = async_sessionmaker(
            bind=engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )
    
    return _session_factory


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency that provides an async database session.
    
    Usage:
        @app.get("/items")
        async def get_items(session: AsyncSession = Depends(get_session)):
            ...
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_session_context() -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager for database sessions outside of FastAPI dependencies.
    
    Usage:
        async with get_session_context() as session:
            result = await session.execute(query)
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_database() -> None:
    """
    Initialize the database by creating all tables.
    
    Should be called during application startup.
    """
    engine = get_engine()
    
    async with engine.begin() as conn:
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)
    
    logger.info("Database tables created successfully")


async def close_database() -> None:
    """
    Close database connections.
    
    Should be called during application shutdown.
    """
    global _engine, _session_factory
    
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("Database connections closed")


async def check_database_health() -> dict:
    """
    Check database connectivity and health.
    
    Returns:
        dict with status, latency, and any error message
    """
    import time
    
    try:
        start = time.monotonic()
        engine = get_engine()
        
        async with engine.connect() as conn:
            await conn.execute("SELECT 1")
        
        latency_ms = int((time.monotonic() - start) * 1000)
        
        return {
            "status": "healthy",
            "latency_ms": latency_ms,
            "message": None,
        }
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "status": "unhealthy",
            "latency_ms": None,
            "message": str(e),
        }


class DatabaseManager:
    """
    Database manager for handling complex transactions and batch operations.
    """
    
    def __init__(self):
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker[AsyncSession]] = None
    
    @property
    def engine(self) -> AsyncEngine:
        if self._engine is None:
            self._engine = get_engine()
        return self._engine
    
    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        if self._session_factory is None:
            self._session_factory = get_session_factory()
        return self._session_factory
    
    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Context manager for database transactions with automatic rollback on error.
        """
        async with self.session_factory() as session:
            async with session.begin():
                yield session
    
    async def execute_in_transaction(self, callback, *args, **kwargs):
        """
        Execute a callback within a transaction.
        
        Args:
            callback: Async function that takes session as first argument
            *args, **kwargs: Additional arguments for the callback
        """
        async with self.transaction() as session:
            return await callback(session, *args, **kwargs)


# Singleton database manager
db_manager = DatabaseManager()
