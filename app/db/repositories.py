"""
Repository layer for data access.

Provides clean abstractions for database operations.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.orm import (
    AgentDefinition,
    AgentMemory,
    AgentState,
    SimulationSession,
    SimulationTurn,
    WhatIfScenario,
)

logger = logging.getLogger(__name__)


class BaseRepository:
    """Base repository with common operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def commit(self) -> None:
        """Commit the current transaction."""
        await self.session.commit()
    
    async def refresh(self, obj: Any) -> None:
        """Refresh an object from the database."""
        await self.session.refresh(obj)


class SessionRepository(BaseRepository):
    """Repository for simulation sessions."""
    
    async def create(
        self,
        scenario: str,
        config: Dict[str, Any],
        name: Optional[str] = None,
        description: Optional[str] = None,
        max_turns: int = 50,
    ) -> SimulationSession:
        """Create a new simulation session."""
        session = SimulationSession(
            scenario=scenario,
            config=config,
            name=name,
            description=description,
            max_turns=max_turns,
            status="active",
            current_turn=0,
        )
        self.session.add(session)
        await self.session.flush()
        await self.session.refresh(session)
        return session
    
    async def get_by_id(
        self,
        session_id: UUID,
        include_turns: bool = False,
        include_agent_states: bool = False,
    ) -> Optional[SimulationSession]:
        """Get a session by ID with optional relationships."""
        query = select(SimulationSession).where(SimulationSession.id == session_id)
        
        if include_turns:
            query = query.options(selectinload(SimulationSession.turns))
        if include_agent_states:
            query = query.options(selectinload(SimulationSession.agent_states))
        
        result = await self.session.execute(query)
        return result.scalar_one_or_none()
    
    async def update_status(
        self,
        session_id: UUID,
        status: str,
        current_turn: Optional[int] = None,
    ) -> bool:
        """Update session status."""
        values = {"status": status, "updated_at": datetime.utcnow()}
        if current_turn is not None:
            values["current_turn"] = current_turn
        
        stmt = (
            update(SimulationSession)
            .where(SimulationSession.id == session_id)
            .values(**values)
        )
        result = await self.session.execute(stmt)
        return result.rowcount > 0
    
    async def increment_turn(self, session_id: UUID) -> int:
        """Increment the current turn and return new value."""
        stmt = (
            update(SimulationSession)
            .where(SimulationSession.id == session_id)
            .values(
                current_turn=SimulationSession.current_turn + 1,
                updated_at=datetime.utcnow(),
            )
            .returning(SimulationSession.current_turn)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one()
    
    async def list_sessions(
        self,
        status: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[SimulationSession]:
        """List sessions with optional status filter."""
        query = select(SimulationSession).order_by(
            SimulationSession.created_at.desc()
        )
        
        if status:
            query = query.where(SimulationSession.status == status)
        
        query = query.limit(limit).offset(offset)
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def delete_expired(self, retention_days: int = 30) -> int:
        """Delete sessions older than retention period."""
        cutoff = datetime.utcnow() - timedelta(days=retention_days)
        stmt = delete(SimulationSession).where(
            SimulationSession.created_at < cutoff
        )
        result = await self.session.execute(stmt)
        return result.rowcount


class TurnRepository(BaseRepository):
    """Repository for simulation turns."""
    
    async def create(
        self,
        session_id: UUID,
        turn_number: int,
        input_context: Optional[str] = None,
    ) -> SimulationTurn:
        """Create a new turn."""
        turn = SimulationTurn(
            session_id=session_id,
            turn_number=turn_number,
            input_context=input_context,
            status="pending",
        )
        self.session.add(turn)
        await self.session.flush()
        return turn
    
    async def update(
        self,
        turn_id: UUID,
        coordinator_summary: Optional[str] = None,
        agent_responses: Optional[List[Dict]] = None,
        duration_ms: Optional[int] = None,
        token_usage: Optional[Dict] = None,
        status: str = "completed",
        error_message: Optional[str] = None,
    ) -> bool:
        """Update turn with results."""
        values = {"status": status, "updated_at": datetime.utcnow()}
        
        if coordinator_summary:
            values["coordinator_summary"] = coordinator_summary
        if agent_responses is not None:
            values["agent_responses"] = agent_responses
        if duration_ms is not None:
            values["duration_ms"] = duration_ms
        if token_usage is not None:
            values["token_usage"] = token_usage
        if error_message:
            values["error_message"] = error_message
        
        stmt = (
            update(SimulationTurn)
            .where(SimulationTurn.id == turn_id)
            .values(**values)
        )
        result = await self.session.execute(stmt)
        return result.rowcount > 0
    
    async def get_by_session(
        self,
        session_id: UUID,
        limit: Optional[int] = None,
    ) -> List[SimulationTurn]:
        """Get all turns for a session."""
        query = (
            select(SimulationTurn)
            .where(SimulationTurn.session_id == session_id)
            .order_by(SimulationTurn.turn_number)
        )
        
        if limit:
            query = query.limit(limit)
        
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def get_latest(self, session_id: UUID) -> Optional[SimulationTurn]:
        """Get the latest turn for a session."""
        query = (
            select(SimulationTurn)
            .where(SimulationTurn.session_id == session_id)
            .order_by(SimulationTurn.turn_number.desc())
            .limit(1)
        )
        result = await self.session.execute(query)
        return result.scalar_one_or_none()


class AgentStateRepository(BaseRepository):
    """Repository for agent states."""
    
    async def create_or_update(
        self,
        session_id: UUID,
        agent_role: str,
        current_focus: Optional[str] = None,
        goals: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentState:
        """Create or update agent state."""
        # Try to get existing state
        query = select(AgentState).where(
            AgentState.session_id == session_id,
            AgentState.agent_role == agent_role,
        )
        result = await self.session.execute(query)
        state = result.scalar_one_or_none()
        
        if state:
            # Update existing
            if current_focus is not None:
                state.current_focus = current_focus
            if goals is not None:
                state.goals = goals
            if context is not None:
                state.context = {**state.context, **context}
            state.updated_at = datetime.utcnow()
        else:
            # Create new
            state = AgentState(
                session_id=session_id,
                agent_role=agent_role,
                current_focus=current_focus,
                goals=goals or [],
                context=context or {},
            )
            self.session.add(state)
        
        await self.session.flush()
        return state
    
    async def increment_messages(
        self,
        session_id: UUID,
        agent_role: str,
        turn_number: int,
    ) -> None:
        """Increment message count for an agent."""
        stmt = (
            update(AgentState)
            .where(
                AgentState.session_id == session_id,
                AgentState.agent_role == agent_role,
            )
            .values(
                messages_sent=AgentState.messages_sent + 1,
                last_active_turn=turn_number,
                updated_at=datetime.utcnow(),
            )
        )
        await self.session.execute(stmt)
    
    async def get_by_session(self, session_id: UUID) -> List[AgentState]:
        """Get all agent states for a session."""
        query = (
            select(AgentState)
            .where(AgentState.session_id == session_id)
            .order_by(AgentState.agent_role)
        )
        result = await self.session.execute(query)
        return list(result.scalars().all())


class AgentDefinitionRepository(BaseRepository):
    """Repository for agent definitions."""
    
    async def get_all_active(self) -> List[AgentDefinition]:
        """Get all active agent definitions."""
        query = (
            select(AgentDefinition)
            .where(AgentDefinition.is_active == True)
            .order_by(AgentDefinition.priority.desc())
        )
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def get_by_role(self, role: str) -> Optional[AgentDefinition]:
        """Get agent definition by role."""
        query = select(AgentDefinition).where(AgentDefinition.role == role)
        result = await self.session.execute(query)
        return result.scalar_one_or_none()
    
    async def create(
        self,
        role: str,
        name: str,
        system_prompt: str,
        description: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
        personality_traits: Optional[Dict[str, float]] = None,
        priority: int = 0,
    ) -> AgentDefinition:
        """Create a new agent definition."""
        agent = AgentDefinition(
            role=role,
            name=name,
            description=description,
            system_prompt=system_prompt,
            capabilities=capabilities or [],
            personality_traits=personality_traits or {},
            priority=priority,
        )
        self.session.add(agent)
        await self.session.flush()
        return agent
    
    async def seed_defaults(self) -> List[AgentDefinition]:
        """Seed default agent definitions if none exist."""
        existing = await self.get_all_active()
        if existing:
            return existing
        
        from app.agents.roles import DEFAULT_AGENTS
        
        agents = []
        for agent_data in DEFAULT_AGENTS:
            agent = await self.create(**agent_data)
            agents.append(agent)
        
        return agents


class MemoryRepository(BaseRepository):
    """Repository for agent memories."""
    
    async def create(
        self,
        session_id: UUID,
        agent_role: str,
        memory_type: str,
        key: str,
        content: Dict[str, Any],
        importance: float = 0.5,
        expires_at: Optional[datetime] = None,
    ) -> AgentMemory:
        """Create a new memory entry."""
        memory = AgentMemory(
            session_id=session_id,
            agent_role=agent_role,
            memory_type=memory_type,
            key=key,
            content=content,
            importance=importance,
            expires_at=expires_at,
        )
        self.session.add(memory)
        await self.session.flush()
        return memory
    
    async def get_by_session(
        self,
        session_id: UUID,
        agent_role: Optional[str] = None,
        memory_type: Optional[str] = None,
    ) -> List[AgentMemory]:
        """Get memories for a session."""
        query = select(AgentMemory).where(AgentMemory.session_id == session_id)
        
        if agent_role:
            query = query.where(AgentMemory.agent_role == agent_role)
        if memory_type:
            query = query.where(AgentMemory.memory_type == memory_type)
        
        query = query.order_by(
            AgentMemory.importance.desc(),
            AgentMemory.created_at.desc(),
        )
        
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def get_relevant(
        self,
        session_id: UUID,
        agent_role: str,
        limit: int = 10,
    ) -> List[AgentMemory]:
        """Get most relevant memories for an agent."""
        now = datetime.utcnow()
        query = (
            select(AgentMemory)
            .where(
                AgentMemory.session_id == session_id,
                AgentMemory.agent_role == agent_role,
                (AgentMemory.expires_at == None) | (AgentMemory.expires_at > now),
            )
            .order_by(
                AgentMemory.importance.desc(),
                AgentMemory.access_count.desc(),
            )
            .limit(limit)
        )
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def update_access(self, memory_id: UUID) -> None:
        """Update access count and timestamp."""
        stmt = (
            update(AgentMemory)
            .where(AgentMemory.id == memory_id)
            .values(
                access_count=AgentMemory.access_count + 1,
                last_accessed_at=datetime.utcnow(),
            )
        )
        await self.session.execute(stmt)
    
    async def delete_expired(self) -> int:
        """Delete expired memories."""
        now = datetime.utcnow()
        stmt = delete(AgentMemory).where(AgentMemory.expires_at < now)
        result = await self.session.execute(stmt)
        return result.rowcount


class WhatIfRepository(BaseRepository):
    """Repository for what-if scenarios."""
    
    async def create(
        self,
        session_id: UUID,
        base_turn: int,
        modifications: List[Dict[str, Any]],
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> WhatIfScenario:
        """Create a new what-if scenario."""
        scenario = WhatIfScenario(
            session_id=session_id,
            base_turn=base_turn,
            name=name,
            description=description,
            modifications=modifications,
            status="pending",
        )
        self.session.add(scenario)
        await self.session.flush()
        return scenario
    
    async def update_results(
        self,
        scenario_id: UUID,
        predicted_outcomes: List[Dict[str, Any]],
        agent_analyses: List[Dict[str, Any]],
        confidence_scores: Dict[str, float],
        status: str = "completed",
    ) -> bool:
        """Update scenario with analysis results."""
        stmt = (
            update(WhatIfScenario)
            .where(WhatIfScenario.id == scenario_id)
            .values(
                predicted_outcomes=predicted_outcomes,
                agent_analyses=agent_analyses,
                confidence_scores=confidence_scores,
                status=status,
                updated_at=datetime.utcnow(),
            )
        )
        result = await self.session.execute(stmt)
        return result.rowcount > 0
    
    async def get_by_session(self, session_id: UUID) -> List[WhatIfScenario]:
        """Get all what-if scenarios for a session."""
        query = (
            select(WhatIfScenario)
            .where(WhatIfScenario.session_id == session_id)
            .order_by(WhatIfScenario.created_at.desc())
        )
        result = await self.session.execute(query)
        return list(result.scalars().all())
