"""
Core package.

Contains the agent engine, coordinator, and simulation controller.
"""

from app.core.agent_engine import (
    AgentEngine,
    Coordinator,
    SimulationState,
    TurnResult,
    create_engine,
    create_coordinator,
)
from app.core.coordinator import (
    SimulationCoordinator,
    create_simulation_coordinator,
)
from app.core.simulation import router as simulation_router

__all__ = [
    # Agent Engine
    "AgentEngine",
    "Coordinator",
    "SimulationState",
    "TurnResult",
    "create_engine",
    "create_coordinator",
    # Extended Coordinator
    "SimulationCoordinator",
    "create_simulation_coordinator",
    # API Router
    "simulation_router",
]
