"""
Database package.

Provides database connection, session management, and repositories.
"""

from app.db.database import (
    get_engine,
    get_session,
    get_session_context,
    get_session_factory,
    init_database,
    close_database,
    check_database_health,
    db_manager,
)
from app.db.repositories import (
    SessionRepository,
    TurnRepository,
    AgentStateRepository,
    AgentDefinitionRepository,
    MemoryRepository,
    WhatIfRepository,
)

__all__ = [
    # Database utilities
    "get_engine",
    "get_session",
    "get_session_context",
    "get_session_factory",
    "init_database",
    "close_database",
    "check_database_health",
    "db_manager",
    # Repositories
    "SessionRepository",
    "TurnRepository",
    "AgentStateRepository",
    "AgentDefinitionRepository",
    "MemoryRepository",
    "WhatIfRepository",
]
