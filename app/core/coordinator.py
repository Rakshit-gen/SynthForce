"""
Coordinator module for high-level simulation orchestration.

Provides the main interface for managing multi-agent simulations.
"""

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from app.core.agent_engine import (
    AgentEngine,
    Coordinator as BaseCoordinator,
    SimulationState,
    TurnResult,
    create_engine,
)
from app.services.groq_client import GroqClient

logger = logging.getLogger(__name__)


class SimulationCoordinator(BaseCoordinator):
    """
    Extended coordinator with additional features.
    
    Provides:
    - Session management
    - What-if analysis coordination
    - Analytics collection
    - State persistence hooks
    """
    
    def __init__(
        self,
        engine: Optional[AgentEngine] = None,
        llm_client: Optional[GroqClient] = None,
        persist_state: bool = True,
    ):
        super().__init__(engine=engine, llm_client=llm_client)
        self.persist_state = persist_state
        self._analytics: Dict[UUID, Dict[str, Any]] = {}
    
    async def start_with_first_turn(
        self,
        session_id: UUID,
        scenario: str,
        max_turns: int = 50,
        agent_roles: Optional[List[str]] = None,
        initial_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Start a simulation and execute the first turn.
        
        Args:
            session_id: Unique session identifier
            scenario: The simulation scenario
            max_turns: Maximum number of turns
            agent_roles: Optional list of agent roles
            initial_prompt: Optional initial user input
            
        Returns:
            Combined start and first turn result
        """
        # Initialize
        state = await self.start_simulation(
            session_id=session_id,
            scenario=scenario,
            max_turns=max_turns,
            agent_roles=agent_roles,
        )
        
        # Initialize analytics
        self._analytics[session_id] = {
            "total_turns": 0,
            "total_tokens": 0,
            "total_duration_ms": 0,
            "agent_participation": {},
        }
        
        # Execute first turn
        result = await self.advance_simulation(
            session_id=session_id,
            user_input=initial_prompt,
        )
        
        # Update analytics
        self._update_analytics(session_id, result)
        
        return {
            "session_id": session_id,
            "state": state,
            "first_turn": result,
            "status": "active",
        }
    
    async def advance_simulation(
        self,
        session_id: UUID,
        user_input: Optional[str] = None,
        focus_agents: Optional[List[str]] = None,
    ) -> TurnResult:
        """
        Advance simulation with extended features.
        
        Args:
            session_id: Session identifier
            user_input: Optional user input
            focus_agents: Optional agent focus list
            
        Returns:
            Turn result with additional metadata
        """
        if session_id not in self._active_simulations:
            raise ValueError(f"No active simulation: {session_id}")
        
        state = self._active_simulations[session_id]
        
        # Execute turn
        result = await self.engine.execute_turn(
            state=state,
            user_input=user_input,
            focus_agents=focus_agents,
        )
        
        # Update analytics
        self._update_analytics(session_id, result)
        
        return result
    
    async def analyze_what_if(
        self,
        session_id: UUID,
        modifications: List[Dict[str, Any]],
        base_turn: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Perform what-if analysis on a simulation.
        
        Args:
            session_id: Session to analyze
            modifications: List of modifications
            base_turn: Turn to base analysis on
            
        Returns:
            What-if analysis results
        """
        state = self.get_simulation_state(session_id)
        if not state:
            raise ValueError(f"No simulation found: {session_id}")
        
        base = base_turn or state.current_turn
        
        # Get agent analyses
        analyses = []
        from app.agents import AgentFactory
        
        for role in state.active_agents:
            agent = AgentFactory.create(role)
            analysis = await agent.analyze_what_if(
                scenario=state.scenario,
                base_turn=base,
                modifications=modifications,
                llm_client=self.engine.llm,
            )
            analyses.append(analysis)
        
        # Synthesize results
        synthesis = await self._synthesize_what_if(analyses, modifications)
        
        return {
            "session_id": session_id,
            "base_turn": base,
            "modifications": modifications,
            "agent_analyses": analyses,
            "synthesis": synthesis,
        }
    
    def get_analytics(self, session_id: UUID) -> Optional[Dict[str, Any]]:
        """Get analytics for a session."""
        return self._analytics.get(session_id)
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Get summary of all active sessions."""
        return [
            {
                "session_id": sid,
                "current_turn": state.current_turn,
                "max_turns": state.max_turns,
                "status": state.status,
                "agents": state.active_agents,
            }
            for sid, state in self._active_simulations.items()
        ]
    
    def _update_analytics(
        self,
        session_id: UUID,
        result: TurnResult,
    ) -> None:
        """Update analytics with turn results."""
        if session_id not in self._analytics:
            self._analytics[session_id] = {
                "total_turns": 0,
                "total_tokens": 0,
                "total_duration_ms": 0,
                "agent_participation": {},
            }
        
        analytics = self._analytics[session_id]
        analytics["total_turns"] += 1
        analytics["total_tokens"] += result.total_tokens
        analytics["total_duration_ms"] += result.duration_ms
        
        for response in result.agent_responses:
            role = response.agent_role
            if role not in analytics["agent_participation"]:
                analytics["agent_participation"][role] = 0
            analytics["agent_participation"][role] += 1
    
    async def _synthesize_what_if(
        self,
        analyses: List[Dict[str, Any]],
        modifications: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Synthesize what-if analyses into overall assessment."""
        # Calculate aggregate confidence
        confidences = [a.get("confidence", 0.5) for a in analyses]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        
        # Collect key points
        all_concerns = []
        all_opportunities = []
        all_actions = []
        
        for analysis in analyses:
            all_concerns.extend(analysis.get("concerns", []))
            all_opportunities.extend(analysis.get("opportunities", []))
            all_actions.extend(analysis.get("recommended_actions", []))
        
        # Generate recommendation
        if avg_confidence > 0.7 and len(all_opportunities) > len(all_concerns):
            recommendation = "proceed"
            reasoning = "High confidence with more opportunities than concerns"
        elif avg_confidence < 0.4 or len(all_concerns) > len(all_opportunities) * 2:
            recommendation = "reconsider"
            reasoning = "Low confidence or significant concerns identified"
        else:
            recommendation = "evaluate"
            reasoning = "Mixed signals require further analysis"
        
        return {
            "overall_confidence": avg_confidence,
            "recommendation": recommendation,
            "reasoning": reasoning,
            "key_concerns": all_concerns[:5],
            "key_opportunities": all_opportunities[:5],
            "recommended_actions": all_actions[:5],
        }


# Factory function
def create_simulation_coordinator(
    llm_client: Optional[GroqClient] = None,
    **engine_kwargs,
) -> SimulationCoordinator:
    """Create a simulation coordinator with custom configuration."""
    engine = create_engine(llm_client=llm_client, **engine_kwargs)
    return SimulationCoordinator(engine=engine)
