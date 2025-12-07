"""
Simulation Controller - Main REST API implementation.

Implements all simulation endpoints:
- /simulate/start
- /simulate/next
- /simulate/what-if
- /agents/list
- /memory/{sessionId}
"""

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.models.schemas import (
    AgentListResponse,
    AgentDefinitionResponse,
    AgentRole,
    MemoryResponse,
    SimulationStartRequest,
    SimulationStartResponse,
    SimulationNextRequest,
    SimulationNextResponse,
    SimulationStateResponse,
    WhatIfRequest,
    WhatIfResponse,
    ErrorResponse,
)
from app.services import SimulationService, get_groq_client

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["simulation"])


# =============================================================================
# Dependencies
# =============================================================================

async def get_simulation_service(
    db: AsyncSession = Depends(get_session),
) -> SimulationService:
    """Dependency to get simulation service."""
    groq_client = get_groq_client()
    return SimulationService(db_session=db, llm_client=groq_client)


# =============================================================================
# Simulation Endpoints
# =============================================================================

@router.post(
    "/simulate/start",
    response_model=SimulationStartResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal error"},
    },
)
async def start_simulation(
    request: SimulationStartRequest,
    service: SimulationService = Depends(get_simulation_service),
) -> SimulationStartResponse:
    """
    Start a new simulation session.
    
    Creates a new simulation with the specified scenario and configuration.
    Returns the session ID and initial state.
    
    **Example Request:**
    ```json
    {
        "scenario": "We need to launch a new product feature within 3 months...",
        "name": "Q2 Feature Launch Planning",
        "config": {
            "max_turns": 20,
            "agents": ["ceo", "pm", "engineering_lead", "designer"]
        }
    }
    ```
    """
    try:
        result = await service.create_session(
            scenario=request.scenario,
            config=request.config,
            name=request.name,
            description=request.description,
            initial_context=request.initial_context,
        )
        
        return SimulationStartResponse(**result)
        
    except Exception as e:
        logger.error(f"Failed to start simulation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.post(
    "/simulate/next",
    response_model=SimulationNextResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        404: {"model": ErrorResponse, "description": "Session not found"},
        500: {"model": ErrorResponse, "description": "Internal error"},
    },
)
async def execute_next_turn(
    request: SimulationNextRequest,
    service: SimulationService = Depends(get_simulation_service),
) -> SimulationNextResponse:
    """
    Execute the next turn in a simulation.
    
    Advances the simulation by one turn, with optional user input
    and agent focus specification.
    
    **Example Request:**
    ```json
    {
        "session_id": "550e8400-e29b-41d4-a716-446655440000",
        "user_input": "Let's focus on the technical feasibility first",
        "focus_agents": ["engineering_lead", "designer"]
    }
    ```
    """
    try:
        result = await service.execute_turn(
            session_id=request.session_id,
            user_input=request.user_input,
            focus_agents=request.focus_agents,
            context_override=request.context_override,
        )
        
        return SimulationNextResponse(**result)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Failed to execute turn: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.post(
    "/simulate/what-if",
    response_model=WhatIfResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        404: {"model": ErrorResponse, "description": "Session not found"},
        500: {"model": ErrorResponse, "description": "Internal error"},
    },
)
async def analyze_what_if(
    request: WhatIfRequest,
    service: SimulationService = Depends(get_simulation_service),
) -> WhatIfResponse:
    """
    Perform what-if scenario analysis.
    
    Analyzes how proposed modifications would affect simulation outcomes.
    Each agent provides their perspective on the changes.
    
    **Example Request:**
    ```json
    {
        "session_id": "550e8400-e29b-41d4-a716-446655440000",
        "modifications": [
            {
                "type": "context",
                "target": "budget",
                "change": {"amount": 500000, "currency": "USD"},
                "description": "Increase budget by 50%"
            }
        ],
        "num_turns_to_simulate": 3
    }
    ```
    """
    try:
        result = await service.analyze_what_if(
            session_id=request.session_id,
            modifications=[m.model_dump() for m in request.modifications],
            base_turn=request.base_turn,
            name=request.name,
            description=request.description,
            num_turns=request.num_turns_to_simulate,
        )
        
        return WhatIfResponse(**result)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Failed to analyze what-if: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.get(
    "/simulate/{session_id}",
    response_model=SimulationStateResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Session not found"},
    },
)
async def get_simulation_state(
    session_id: UUID,
    service: SimulationService = Depends(get_simulation_service),
) -> SimulationStateResponse:
    """
    Get the current state of a simulation session.
    
    Returns full session state including agent states and configuration.
    """
    try:
        result = await service.get_session_state(session_id)
        return SimulationStateResponse(**result)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


# =============================================================================
# Agent Endpoints
# =============================================================================

@router.get(
    "/agents/list",
    response_model=AgentListResponse,
)
async def list_agents(
    active_only: bool = Query(True, description="Only return active agents"),
) -> AgentListResponse:
    """
    List all available agents.
    
    Returns the list of agent types that can participate in simulations,
    along with their capabilities and descriptions.
    """
    from app.agents.roles import DEFAULT_AGENTS
    
    agents = []
    for agent_data in DEFAULT_AGENTS:
        if not active_only or agent_data.get("is_active", True):
            agents.append(AgentDefinitionResponse(
                id=UUID("00000000-0000-0000-0000-000000000000"),  # Placeholder
                role=AgentRole(agent_data["role"]),
                name=agent_data["name"],
                description=agent_data.get("description"),
                capabilities=agent_data.get("capabilities", []),
                personality_traits=agent_data.get("personality_traits", {}),
                priority=agent_data.get("priority", 0),
                is_active=True,
            ))
    
    return AgentListResponse(
        agents=agents,
        total=len(agents),
    )


@router.get(
    "/agents/{role}",
    response_model=AgentDefinitionResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Agent not found"},
    },
)
async def get_agent(role: AgentRole) -> AgentDefinitionResponse:
    """
    Get details for a specific agent role.
    """
    from app.agents.roles import get_agent_by_role
    
    try:
        agent_data = get_agent_by_role(role.value)
        
        return AgentDefinitionResponse(
            id=UUID("00000000-0000-0000-0000-000000000000"),
            role=role,
            name=agent_data["name"],
            description=agent_data.get("description"),
            capabilities=agent_data.get("capabilities", []),
            personality_traits=agent_data.get("personality_traits", {}),
            priority=agent_data.get("priority", 0),
            is_active=True,
        )
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent role '{role}' not found",
        )


# =============================================================================
# Memory Endpoints
# =============================================================================

@router.get(
    "/memory/{session_id}",
    response_model=MemoryResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Session not found"},
    },
)
async def get_session_memory(
    session_id: UUID,
    agent_role: Optional[AgentRole] = Query(None, description="Filter by agent role"),
    service: SimulationService = Depends(get_simulation_service),
) -> MemoryResponse:
    """
    Get memory/state for a simulation session.
    
    Returns all stored memories for the session, optionally filtered by agent.
    Memory includes context, decisions, and insights from each turn.
    """
    try:
        result = await service.get_session_memory(session_id)
        
        # Filter by agent if specified
        if agent_role:
            filtered_memories = {
                agent_role.value: result["memories_by_agent"].get(agent_role.value, [])
            }
            result["memories_by_agent"] = filtered_memories
            result["total_memories"] = len(filtered_memories.get(agent_role.value, []))
        
        return MemoryResponse(**result)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.delete(
    "/memory/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        404: {"model": ErrorResponse, "description": "Session not found"},
    },
)
async def clear_session_memory(
    session_id: UUID,
    db: AsyncSession = Depends(get_session),
):
    """
    Clear all memory for a simulation session.
    """
    from sqlalchemy import delete
    from app.models.orm import AgentMemory
    
    # Delete all memories for this session
    stmt = delete(AgentMemory).where(AgentMemory.session_id == session_id)
    await db.execute(stmt)
    await db.commit()


# =============================================================================
# Session Management Endpoints
# =============================================================================

@router.post(
    "/simulate/{session_id}/pause",
    status_code=status.HTTP_200_OK,
)
async def pause_simulation(
    session_id: UUID,
    db: AsyncSession = Depends(get_session),
) -> Dict[str, Any]:
    """Pause a running simulation."""
    from app.db.repositories import SessionRepository
    
    repo = SessionRepository(db)
    success = await repo.update_status(session_id, "paused")
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )
    
    return {"session_id": str(session_id), "status": "paused"}


@router.post(
    "/simulate/{session_id}/resume",
    status_code=status.HTTP_200_OK,
)
async def resume_simulation(
    session_id: UUID,
    db: AsyncSession = Depends(get_session),
) -> Dict[str, Any]:
    """Resume a paused simulation."""
    from app.db.repositories import SessionRepository
    
    repo = SessionRepository(db)
    success = await repo.update_status(session_id, "active")
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )
    
    return {"session_id": str(session_id), "status": "active"}


@router.delete(
    "/simulate/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_simulation(
    session_id: UUID,
    db: AsyncSession = Depends(get_session),
):
    """Delete a simulation session and all related data."""
    from sqlalchemy import delete
    from app.models.orm import SimulationSession
    
    stmt = delete(SimulationSession).where(SimulationSession.id == session_id)
    result = await db.execute(stmt)
    
    if result.rowcount == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )
    
    await db.commit()
