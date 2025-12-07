"""
Unit tests for the Agent Engine.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from app.agents import (
    AgentContext,
    AgentResponse,
    AgentFactory,
    ALL_ROLES,
    get_agent_by_role,
)
from app.core.agent_engine import (
    AgentEngine,
    Coordinator,
    SimulationState,
    TurnResult,
)
from app.services.groq_client import LLMResponse


@pytest.fixture
def mock_llm_response():
    """Create a mock LLM response."""
    return LLMResponse(
        content="This is a test response from the agent.",
        model="llama-3.3-70b-versatile",
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        finish_reason="stop",
        latency_ms=200,
    )


@pytest.fixture
def mock_groq_client(mock_llm_response):
    """Create a mock Groq client."""
    client = AsyncMock()
    client.complete = AsyncMock(return_value=mock_llm_response)
    client.health_check = AsyncMock(return_value={"status": "healthy"})
    return client


@pytest.fixture
def agent_engine(mock_groq_client):
    """Create an agent engine with mock client."""
    return AgentEngine(llm_client=mock_groq_client)


@pytest.fixture
def sample_context():
    """Create a sample agent context."""
    return AgentContext(
        session_id=uuid4(),
        turn_number=1,
        scenario="Test scenario for product launch",
        max_turns=10,
        active_agents=["ceo", "pm", "engineering_lead"],
    )


class TestAgentFactory:
    """Tests for AgentFactory."""
    
    def test_create_single_agent(self):
        """Test creating a single agent."""
        agent = AgentFactory.create("ceo")
        
        assert agent.role == "ceo"
        assert agent.name == "Chief Executive Officer"
        assert len(agent.capabilities) > 0
    
    def test_create_all_agents(self):
        """Test creating all agents."""
        agents = AgentFactory.create_all()
        
        assert len(agents) == len(ALL_ROLES)
        # Verify they're sorted by priority
        priorities = [a.priority for a in agents]
        assert priorities == sorted(priorities, reverse=True)
    
    def test_create_specific_roles(self):
        """Test creating specific roles."""
        roles = ["ceo", "pm"]
        agents = AgentFactory.create_all(roles=roles)
        
        assert len(agents) == 2
        assert set(a.role for a in agents) == set(roles)
    
    def test_agent_caching(self):
        """Test that agents are cached."""
        agent1 = AgentFactory.create("ceo", use_cache=True)
        agent2 = AgentFactory.create("ceo", use_cache=True)
        
        assert agent1 is agent2
    
    def test_invalid_role(self):
        """Test creating an invalid role."""
        with pytest.raises(ValueError):
            get_agent_by_role("invalid_role")


class TestAgentContext:
    """Tests for AgentContext."""
    
    def test_context_creation(self):
        """Test creating an agent context."""
        session_id = uuid4()
        context = AgentContext(
            session_id=session_id,
            turn_number=1,
            scenario="Test",
            max_turns=10,
            active_agents=["ceo"],
        )
        
        assert context.session_id == session_id
        assert context.turn_number == 1
        assert context.scenario == "Test"
    
    def test_context_with_optional_fields(self):
        """Test context with all optional fields."""
        context = AgentContext(
            session_id=uuid4(),
            turn_number=2,
            scenario="Test",
            max_turns=10,
            active_agents=["ceo", "pm"],
            previous_summary="Previous turn summary",
            recent_messages=[{"agent_role": "ceo", "response": "Hello"}],
            memories=[{"key": "test", "content": {}}],
            user_input="User question",
        )
        
        assert context.previous_summary == "Previous turn summary"
        assert len(context.recent_messages) == 1
        assert context.user_input == "User question"


class TestAgentResponse:
    """Tests for AgentResponse."""
    
    def test_response_creation(self):
        """Test creating an agent response."""
        response = AgentResponse(
            agent_role="ceo",
            response="Test response",
        )
        
        assert response.agent_role == "ceo"
        assert response.response == "Test response"
        assert response.confidence == 0.8  # Default
    
    def test_response_to_dict(self):
        """Test converting response to dictionary."""
        response = AgentResponse(
            agent_role="ceo",
            response="Test",
            actions_proposed=["action1"],
            confidence=0.9,
        )
        
        result = response.to_dict()
        
        assert result["agent_role"] == "ceo"
        assert result["response"] == "Test"
        assert result["actions_proposed"] == ["action1"]
        assert result["confidence"] == 0.9


class TestSimulationState:
    """Tests for SimulationState."""
    
    def test_state_creation(self):
        """Test creating simulation state."""
        state = SimulationState(
            session_id=uuid4(),
            scenario="Test scenario",
            current_turn=0,
            max_turns=10,
            active_agents=["ceo", "pm"],
        )
        
        assert state.current_turn == 0
        assert state.status == "active"
        assert not state.is_complete()
    
    def test_is_complete_by_turns(self):
        """Test completion by turn count."""
        state = SimulationState(
            session_id=uuid4(),
            scenario="Test",
            current_turn=10,
            max_turns=10,
            active_agents=["ceo"],
        )
        
        assert state.is_complete()
    
    def test_is_complete_by_status(self):
        """Test completion by status."""
        state = SimulationState(
            session_id=uuid4(),
            scenario="Test",
            current_turn=5,
            max_turns=10,
            active_agents=["ceo"],
            status="completed",
        )
        
        assert state.is_complete()
    
    def test_get_recent_context(self):
        """Test getting recent context."""
        state = SimulationState(
            session_id=uuid4(),
            scenario="Test",
            current_turn=3,
            max_turns=10,
            active_agents=["ceo"],
        )
        
        # Add turn history
        for i in range(3):
            state.turn_history.append(
                TurnResult(
                    turn_number=i + 1,
                    agent_responses=[
                        AgentResponse(agent_role="ceo", response=f"Turn {i+1}")
                    ],
                    coordinator_summary=f"Summary {i+1}",
                    duration_ms=100,
                    total_tokens=100,
                )
            )
        
        context = state.get_recent_context(num_turns=2)
        assert len(context) == 2


class TestAgentEngine:
    """Tests for AgentEngine."""
    
    @pytest.mark.asyncio
    async def test_initialize_simulation(self, agent_engine):
        """Test initializing a simulation."""
        session_id = uuid4()
        
        state = await agent_engine.initialize_simulation(
            session_id=session_id,
            scenario="Test scenario",
            max_turns=10,
            agent_roles=["ceo", "pm"],
        )
        
        assert state.session_id == session_id
        assert state.scenario == "Test scenario"
        assert state.max_turns == 10
        assert len(state.active_agents) == 2
        assert "ceo" in state.agent_states
        assert "pm" in state.agent_states
    
    @pytest.mark.asyncio
    async def test_execute_turn(self, agent_engine, mock_groq_client):
        """Test executing a turn."""
        session_id = uuid4()
        
        state = await agent_engine.initialize_simulation(
            session_id=session_id,
            scenario="Test scenario",
            max_turns=10,
            agent_roles=["ceo", "pm"],
        )
        
        result = await agent_engine.execute_turn(state)
        
        assert result.turn_number == 1
        assert len(result.agent_responses) == 2
        assert result.coordinator_summary is not None
        assert result.duration_ms > 0
        assert state.current_turn == 1
    
    @pytest.mark.asyncio
    async def test_execute_turn_with_user_input(self, agent_engine):
        """Test executing a turn with user input."""
        state = await agent_engine.initialize_simulation(
            session_id=uuid4(),
            scenario="Test",
            max_turns=10,
            agent_roles=["ceo"],
        )
        
        result = await agent_engine.execute_turn(
            state,
            user_input="Focus on budget",
        )
        
        assert result.turn_number == 1
    
    @pytest.mark.asyncio
    async def test_execute_turn_complete_simulation(self, agent_engine):
        """Test that executing on complete simulation raises error."""
        state = SimulationState(
            session_id=uuid4(),
            scenario="Test",
            current_turn=10,
            max_turns=10,
            active_agents=["ceo"],
        )
        
        with pytest.raises(ValueError, match="already complete"):
            await agent_engine.execute_turn(state)
    
    def test_add_hooks(self, agent_engine):
        """Test adding hooks."""
        pre_hook = MagicMock()
        post_hook = MagicMock()
        
        agent_engine.add_pre_turn_hook(pre_hook)
        agent_engine.add_post_turn_hook(post_hook)
        
        assert pre_hook in agent_engine._pre_turn_hooks
        assert post_hook in agent_engine._post_turn_hooks


class TestCoordinator:
    """Tests for Coordinator."""
    
    @pytest.mark.asyncio
    async def test_start_simulation(self, mock_groq_client):
        """Test starting a simulation through coordinator."""
        coordinator = Coordinator(llm_client=mock_groq_client)
        
        session_id = uuid4()
        state = await coordinator.start_simulation(
            session_id=session_id,
            scenario="Test scenario",
            max_turns=10,
        )
        
        assert state.session_id == session_id
        assert session_id in coordinator._active_simulations
    
    @pytest.mark.asyncio
    async def test_advance_simulation(self, mock_groq_client):
        """Test advancing a simulation."""
        coordinator = Coordinator(llm_client=mock_groq_client)
        
        session_id = uuid4()
        await coordinator.start_simulation(
            session_id=session_id,
            scenario="Test",
            max_turns=10,
            agent_roles=["ceo"],
        )
        
        result = await coordinator.advance_simulation(session_id)
        
        assert result.turn_number == 1
    
    @pytest.mark.asyncio
    async def test_advance_invalid_session(self, mock_groq_client):
        """Test advancing invalid session."""
        coordinator = Coordinator(llm_client=mock_groq_client)
        
        with pytest.raises(ValueError, match="No active simulation"):
            await coordinator.advance_simulation(uuid4())
    
    def test_get_simulation_state(self, mock_groq_client):
        """Test getting simulation state."""
        coordinator = Coordinator(llm_client=mock_groq_client)
        
        # No active simulation
        assert coordinator.get_simulation_state(uuid4()) is None
    
    def test_stop_simulation(self, mock_groq_client):
        """Test stopping simulation."""
        coordinator = Coordinator(llm_client=mock_groq_client)
        
        # Stop non-existent
        assert coordinator.stop_simulation(uuid4()) is False
