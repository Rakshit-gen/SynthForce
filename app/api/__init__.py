"""
API routes package.

Re-exports all API routers for convenience.
"""

from app.core.simulation import router as simulation_router

__all__ = ["simulation_router"]
