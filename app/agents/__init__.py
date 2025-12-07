"""
Agents package.

Contains agent definitions, templates, and base classes.
"""

from app.agents.base import (
    AgentContext,
    AgentResponse,
    BaseAgent,
    SimulationAgent,
    AgentFactory,
)
from app.agents.roles import (
    ROLE_CEO,
    ROLE_PM,
    ROLE_ENGINEERING_LEAD,
    ROLE_DESIGNER,
    ROLE_SALES,
    ROLE_SUPPORT,
    ROLE_SIMULATION_ANALYST,
    ALL_ROLES,
    DEFAULT_AGENTS,
    get_agent_by_role,
    get_all_agent_roles,
    get_agent_priority,
    get_turn_order,
)
from app.agents.templates import (
    build_turn_prompt,
    build_what_if_prompt,
    build_synthesis_prompt,
    format_agent_messages,
)

__all__ = [
    # Base classes
    "AgentContext",
    "AgentResponse",
    "BaseAgent",
    "SimulationAgent",
    "AgentFactory",
    # Role constants
    "ROLE_CEO",
    "ROLE_PM",
    "ROLE_ENGINEERING_LEAD",
    "ROLE_DESIGNER",
    "ROLE_SALES",
    "ROLE_SUPPORT",
    "ROLE_SIMULATION_ANALYST",
    "ALL_ROLES",
    "DEFAULT_AGENTS",
    # Role utilities
    "get_agent_by_role",
    "get_all_agent_roles",
    "get_agent_priority",
    "get_turn_order",
    # Template utilities
    "build_turn_prompt",
    "build_what_if_prompt",
    "build_synthesis_prompt",
    "format_agent_messages",
]
