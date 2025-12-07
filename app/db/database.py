"""
Database connection and session management.

Provides async SQLAlchemy engine, session factory, and connection utilities.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

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


def _normalize_database_url(url: str) -> tuple[str, dict]:
    """
    Normalize database URL by converting sslmode to asyncpg-compatible SSL config.
    
    asyncpg doesn't support 'sslmode' query parameter. Instead, it uses 'ssl' 
    parameter in connect_args. This function:
    1. Parses the URL and extracts sslmode
    2. Removes sslmode from query string
    3. Returns normalized URL and SSL config for connect_args
    
    Returns:
        tuple: (normalized_url, connect_args_dict)
    """
    parsed = urlparse(url)
    query_params = parse_qs(parsed.query)
    
    # Extract sslmode if present
    sslmode = None
    if "sslmode" in query_params:
        sslmode = query_params.pop("sslmode")[0].lower()
    
    # Remove sslmode from query string
    new_query = urlencode(query_params, doseq=True)
    normalized_url = urlunparse((
        parsed.scheme,
        parsed.netloc,
        parsed.path,
        parsed.params,
        new_query,
        parsed.fragment
    ))
    
    # Convert sslmode to asyncpg ssl parameter
    connect_args = {}
    if sslmode:
        # asyncpg expects ssl=True/False or an SSL context
        # sslmode values: disable, allow, prefer, require, verify-ca, verify-full
        if sslmode in ("require", "prefer", "verify-ca", "verify-full"):
            connect_args["ssl"] = True
        elif sslmode == "disable":
            connect_args["ssl"] = False
        # For allow, we'll default to False (asyncpg doesn't have exact equivalent)
        elif sslmode == "allow":
            connect_args["ssl"] = False
    
    return normalized_url, connect_args


def get_engine() -> AsyncEngine:
    """
    Get or create the async database engine.
    
    Uses connection pooling for production, NullPool for testing.
    """
    global _engine
    
    if _engine is None:
        settings = get_settings()
        
        # Normalize database URL and extract SSL config
        db_url, connect_args = _normalize_database_url(settings.database.url)
        
        # Engine configuration based on environment
        engine_kwargs = {
            "echo": settings.database.echo or settings.debug,
            "future": True,
        }
        
        # Add SSL configuration if present
        if connect_args:
            engine_kwargs["connect_args"] = connect_args
        
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
            db_url,
            **engine_kwargs
        )
        
        logger.info(
            f"Database engine created",
            extra={
                "database_url": db_url.split("@")[-1] if "@" in db_url else db_url,  # Hide credentials
                "environment": settings.environment,
                "ssl_enabled": connect_args.get("ssl", False) if connect_args else False,
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
