"""
Agent Engine - Core orchestration for multi-agent simulations.

This module provides:
1. LLM wrapper for Groq integration
2. Role templates and agent configuration
3. Turn-based orchestration
4. Memory persistence

This is the heart of the simulation system, coordinating agent interactions
and managing the flow of information between agents.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable, Awaitable
from uuid import UUID

from app.agents import (
    AgentContext,
    AgentFactory,
    AgentResponse,
    ALL_ROLES,
    get_turn_order,
)
from app.services.groq_client import GroqClient, LLMResponse

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TurnResult:
    """Result of a simulation turn."""
    
    turn_number: int
    agent_responses: List[AgentResponse]
    coordinator_summary: str
    duration_ms: int
    total_tokens: int
    status: str = "completed"
    error: Optional[str] = None


@dataclass
class SimulationState:
    """Current state of a simulation."""
    
    session_id: UUID
    scenario: str
    current_turn: int
    max_turns: int
    active_agents: List[str]
    agent_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    turn_history: List[TurnResult] = field(default_factory=list)
    memory_store: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    status: str = "active"
    
    def is_complete(self) -> bool:
        """Check if simulation is complete."""
        return self.current_turn >= self.max_turns or self.status == "completed"
    
    def get_recent_context(self, num_turns: int = 3) -> List[Dict[str, Any]]:
        """Get recent context from turn history."""
        context = []
        for turn in self.turn_history[-num_turns:]:
            for response in turn.agent_responses:
                context.append(response.to_dict())
        return context


# =============================================================================
# Agent Engine
# =============================================================================

class AgentEngine:
    """
    Core engine for multi-agent simulations.
    
    Manages:
    - LLM integration via Groq
    - Agent creation and lifecycle
    - Turn-based orchestration
    - Memory management
    - Coordinator synthesis
    """
    
    def __init__(
        self,
        llm_client: Optional[GroqClient] = None,
        default_temperature: float = 0.7,
        default_max_tokens: int = 1024,
        enable_parallel_execution: bool = False,
    ):
        """
        Initialize the agent engine.
        
        Args:
            llm_client: Optional pre-configured Groq client
            default_temperature: Default temperature for LLM calls
            default_max_tokens: Default max tokens per response
            enable_parallel_execution: Whether to run agents in parallel
        """
        self.llm_client = llm_client
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.enable_parallel = enable_parallel_execution
        
        # Hooks for extensibility
        self._pre_turn_hooks: List[Callable] = []
        self._post_turn_hooks: List[Callable] = []
        self._response_transformers: List[Callable] = []
    
    @property
    def llm(self) -> GroqClient:
        """Get the LLM client, creating if needed."""
        if self.llm_client is None:
            from app.services.groq_client import get_groq_client
            self.llm_client = get_groq_client()
        return self.llm_client
    
    async def initialize_simulation(
        self,
        session_id: UUID,
        scenario: str,
        max_turns: int = 50,
        agent_roles: Optional[List[str]] = None,
        initial_context: Optional[Dict[str, Any]] = None,
    ) -> SimulationState:
        """
        Initialize a new simulation.
        
        Args:
            session_id: Unique session identifier
            scenario: The simulation scenario
            max_turns: Maximum number of turns
            agent_roles: List of agent roles to include
            initial_context: Initial context for agents
            
        Returns:
            Initialized SimulationState
        """
        roles = agent_roles or ALL_ROLES
        
        # Initialize agent states
        agent_states = {}
        for role in roles:
            agent = AgentFactory.create(role)
            agent_states[role] = {
                "name": agent.name,
                "capabilities": agent.capabilities,
                "personality": agent.personality,
                "current_focus": None,
                "goals": [],
                "context": initial_context or {},
            }
        
        state = SimulationState(
            session_id=session_id,
            scenario=scenario,
            current_turn=0,
            max_turns=max_turns,
            active_agents=roles,
            agent_states=agent_states,
        )
        
        logger.info(
            f"Initialized simulation",
            extra={
                "session_id": str(session_id),
                "agents": roles,
                "max_turns": max_turns,
            }
        )
        
        return state
    
    async def execute_turn(
        self,
        state: SimulationState,
        user_input: Optional[str] = None,
        focus_agents: Optional[List[str]] = None,
    ) -> TurnResult:
        """
        Execute a single simulation turn.
        
        Args:
            state: Current simulation state
            user_input: Optional user input for this turn
            focus_agents: Optional subset of agents to execute
            
        Returns:
            TurnResult with agent responses and summary
        """
        import time
        
        if state.is_complete():
            raise ValueError("Simulation is already complete")
        
        start_time = time.monotonic()
        state.current_turn += 1
        
        # Run pre-turn hooks
        await self._run_hooks(self._pre_turn_hooks, state)
        
        # Determine which agents participate
        active_agents = focus_agents or state.active_agents
        
        # Get previous summary for context
        previous_summary = None
        if state.turn_history:
            previous_summary = state.turn_history[-1].coordinator_summary
        
        # Build context
        context = AgentContext(
            session_id=state.session_id,
            turn_number=state.current_turn,
            scenario=state.scenario,
            max_turns=state.max_turns,
            active_agents=active_agents,
            previous_summary=previous_summary,
            recent_messages=state.get_recent_context(),
            user_input=user_input,
        )
        
        # Execute agent turns
        if self.enable_parallel:
            responses = await self._execute_parallel(context, active_agents)
        else:
            responses = await self._execute_sequential(context, active_agents)
        
        # Apply response transformers
        for transformer in self._response_transformers:
            responses = await transformer(responses)
        
        # Generate coordinator summary
        summary = await self._generate_coordinator_summary(responses)
        
        # Calculate metrics
        duration_ms = int((time.monotonic() - start_time) * 1000)
        total_tokens = sum(
            r.token_usage.get("total_tokens", 0) for r in responses
        )
        
        # Create turn result
        result = TurnResult(
            turn_number=state.current_turn,
            agent_responses=responses,
            coordinator_summary=summary,
            duration_ms=duration_ms,
            total_tokens=total_tokens,
        )
        
        # Update state
        state.turn_history.append(result)
        
        # Store memories
        await self._store_memories(state, result)
        
        # Run post-turn hooks
        await self._run_hooks(self._post_turn_hooks, state, result)
        
        logger.info(
            f"Completed turn {state.current_turn}",
            extra={
                "session_id": str(state.session_id),
                "duration_ms": duration_ms,
                "tokens": total_tokens,
            }
        )
        
        return result
    
    async def _execute_sequential(
        self,
        context: AgentContext,
        agent_roles: List[str],
    ) -> List[AgentResponse]:
        """Execute agents sequentially in priority order."""
        responses = []
        
        # Sort by priority
        ordered_roles = sorted(
            agent_roles,
            key=lambda r: AgentFactory.create(r).priority,
            reverse=True,
        )
        
        for role in ordered_roles:
            agent = AgentFactory.create(role)
            
            # Add previous responses to context
            context.recent_messages = [r.to_dict() for r in responses]
            
            # Get relevant memories
            memories = self._get_relevant_memories(context.session_id, role)
            context.memories = memories
            
            # Generate response
            response = await agent.generate_response(context, self.llm)
            responses.append(response)
        
        return responses
    
    async def _execute_parallel(
        self,
        context: AgentContext,
        agent_roles: List[str],
    ) -> List[AgentResponse]:
        """Execute agents in parallel."""
        async def execute_agent(role: str) -> AgentResponse:
            agent = AgentFactory.create(role)
            
            # Create copy of context for this agent
            agent_context = AgentContext(
                session_id=context.session_id,
                turn_number=context.turn_number,
                scenario=context.scenario,
                max_turns=context.max_turns,
                active_agents=context.active_agents,
                previous_summary=context.previous_summary,
                recent_messages=context.recent_messages.copy(),
                user_input=context.user_input,
                memories=self._get_relevant_memories(context.session_id, role),
            )
            
            return await agent.generate_response(agent_context, self.llm)
        
        # Execute all agents in parallel
        tasks = [execute_agent(role) for role in agent_roles]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_responses = []
        for r in responses:
            if isinstance(r, Exception):
                logger.error(f"Agent execution failed: {r}")
            else:
                valid_responses.append(r)
        
        return valid_responses
    
    async def _generate_coordinator_summary(
        self,
        responses: List[AgentResponse],
    ) -> str:
        """Generate a synthesis of agent responses."""
        from app.agents.templates import build_synthesis_prompt
        
        prompt = build_synthesis_prompt([r.to_dict() for r in responses])
        
        result = await self.llm.complete(
            system_prompt="You are a simulation coordinator. Synthesize the team's discussion into a clear summary.",
            user_prompt=prompt,
            temperature=0.5,
            max_tokens=500,
        )
        
        return result.content
    
    async def _store_memories(
        self,
        state: SimulationState,
        result: TurnResult,
    ) -> None:
        """Store memories from the turn."""
        for response in result.agent_responses:
            role = response.agent_role
            
            if role not in state.memory_store:
                state.memory_store[role] = []
            
            # Store response summary as memory
            state.memory_store[role].append({
                "turn": result.turn_number,
                "type": "response",
                "content": response.response[:500],
                "actions": response.actions_proposed,
                "timestamp": datetime.utcnow().isoformat(),
            })
            
            # Limit memory size
            if len(state.memory_store[role]) > 20:
                state.memory_store[role] = state.memory_store[role][-20:]
    
    def _get_relevant_memories(
        self,
        session_id: UUID,
        agent_role: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Get relevant memories for an agent (from state)."""
        # This would typically query a database
        # For in-memory operation, return empty list
        return []
    
    async def _run_hooks(
        self,
        hooks: List[Callable],
        *args,
    ) -> None:
        """Run registered hooks."""
        for hook in hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(*args)
                else:
                    hook(*args)
            except Exception as e:
                logger.error(f"Hook execution failed: {e}")
    
    def add_pre_turn_hook(self, hook: Callable) -> None:
        """Add a hook to run before each turn."""
        self._pre_turn_hooks.append(hook)
    
    def add_post_turn_hook(self, hook: Callable) -> None:
        """Add a hook to run after each turn."""
        self._post_turn_hooks.append(hook)
    
    def add_response_transformer(self, transformer: Callable) -> None:
        """Add a transformer to modify responses."""
        self._response_transformers.append(transformer)


# =============================================================================
# Coordinator
# =============================================================================

class Coordinator:
    """
    High-level coordinator for simulation management.
    
    Provides a simplified interface for running simulations.
    """
    
    def __init__(
        self,
        engine: Optional[AgentEngine] = None,
        llm_client: Optional[GroqClient] = None,
    ):
        self.engine = engine or AgentEngine(llm_client=llm_client)
        self._active_simulations: Dict[UUID, SimulationState] = {}
    
    async def start_simulation(
        self,
        session_id: UUID,
        scenario: str,
        max_turns: int = 50,
        agent_roles: Optional[List[str]] = None,
    ) -> SimulationState:
        """Start a new simulation."""
        state = await self.engine.initialize_simulation(
            session_id=session_id,
            scenario=scenario,
            max_turns=max_turns,
            agent_roles=agent_roles,
        )
        
        self._active_simulations[session_id] = state
        return state
    
    async def advance_simulation(
        self,
        session_id: UUID,
        user_input: Optional[str] = None,
    ) -> TurnResult:
        """Advance a simulation by one turn."""
        if session_id not in self._active_simulations:
            raise ValueError(f"No active simulation with ID {session_id}")
        
        state = self._active_simulations[session_id]
        return await self.engine.execute_turn(state, user_input)
    
    async def run_full_simulation(
        self,
        session_id: UUID,
        scenario: str,
        max_turns: int = 10,
        agent_roles: Optional[List[str]] = None,
    ) -> List[TurnResult]:
        """Run a complete simulation."""
        state = await self.start_simulation(
            session_id=session_id,
            scenario=scenario,
            max_turns=max_turns,
            agent_roles=agent_roles,
        )
        
        results = []
        while not state.is_complete():
            result = await self.engine.execute_turn(state)
            results.append(result)
        
        return results
    
    def get_simulation_state(self, session_id: UUID) -> Optional[SimulationState]:
        """Get the current state of a simulation."""
        return self._active_simulations.get(session_id)
    
    def stop_simulation(self, session_id: UUID) -> bool:
        """Stop a running simulation."""
        if session_id in self._active_simulations:
            self._active_simulations[session_id].status = "stopped"
            return True
        return False


# =============================================================================
# Factory Functions
# =============================================================================

def create_engine(
    llm_client: Optional[GroqClient] = None,
    **kwargs,
) -> AgentEngine:
    """Create and configure an agent engine."""
    return AgentEngine(llm_client=llm_client, **kwargs)


def create_coordinator(
    engine: Optional[AgentEngine] = None,
    llm_client: Optional[GroqClient] = None,
) -> Coordinator:
    """Create a simulation coordinator."""
    return Coordinator(engine=engine, llm_client=llm_client)
