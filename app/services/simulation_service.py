"""
Simulation service layer.

Provides business logic for simulation operations.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.agents import AgentContext, AgentFactory, AgentResponse
from app.agents.templates import build_synthesis_prompt
from app.db.repositories import (
    AgentStateRepository,
    MemoryRepository,
    SessionRepository,
    TurnRepository,
    WhatIfRepository,
)
from app.models.schemas import (
    AgentRole,
    MemoryType,
    SimulationConfig,
    SessionStatus,
    TurnStatus,
)
from app.services.groq_client import GroqClient
from app.services.function_point_analyzer import FunctionPointAnalyzer
from app.services.project_scheduler import ProjectScheduler

logger = logging.getLogger(__name__)


class SimulationService:
    """
    Service for managing simulation operations.
    
    Handles session creation, turn execution, and what-if analysis.
    """
    
    def __init__(
        self,
        db_session: AsyncSession,
        llm_client: GroqClient,
    ):
        self.db = db_session
        self.llm = llm_client
        
        # Initialize repositories
        self.sessions = SessionRepository(db_session)
        self.turns = TurnRepository(db_session)
        self.agent_states = AgentStateRepository(db_session)
        self.memories = MemoryRepository(db_session)
        self.what_ifs = WhatIfRepository(db_session)
        
        # Initialize analysis services
        self.fp_analyzer = FunctionPointAnalyzer()
        self.scheduler = ProjectScheduler()
    
    async def create_session(
        self,
        scenario: str,
        config: SimulationConfig,
        name: Optional[str] = None,
        description: Optional[str] = None,
        initial_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a new simulation session.
        
        Args:
            scenario: The simulation scenario
            config: Simulation configuration
            name: Optional session name
            description: Optional description
            initial_context: Optional initial context for agents
            
        Returns:
            Session creation result
        """
        # Create session record
        session = await self.sessions.create(
            scenario=scenario,
            config=config.model_dump(),
            name=name,
            description=description,
            max_turns=config.max_turns,
        )
        
        # Initialize agent states
        agent_roles = [role.value for role in config.agents]
        for role in agent_roles:
            await self.agent_states.create_or_update(
                session_id=session.id,
                agent_role=role,
                goals=[],
                context=initial_context or {},
            )
        
        await self.db.commit()
        
        logger.info(
            f"Created simulation session",
            extra={
                "session_id": str(session.id),
                "agents": agent_roles,
            }
        )
        
        return {
            "session_id": session.id,
            "status": SessionStatus.ACTIVE,
            "scenario": scenario,
            "config": config,
            "agents": agent_roles,
            "message": "Simulation session created successfully",
            "created_at": session.created_at,
        }
    
    async def execute_turn(
        self,
        session_id: UUID,
        user_input: Optional[str] = None,
        focus_agents: Optional[List[AgentRole]] = None,
        context_override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the next turn in a simulation.
        
        Args:
            session_id: The session ID
            user_input: Optional user input for this turn
            focus_agents: Optional list of agents to include
            context_override: Optional context overrides
            
        Returns:
            Turn execution results
        """
        import time
        
        start_time = time.monotonic()
        
        # Get session
        session = await self.sessions.get_by_id(
            session_id,
            include_turns=True,
            include_agent_states=True,
        )
        
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        if session.status != "active":
            raise ValueError(f"Session is not active (status: {session.status})")
        
        if session.current_turn >= session.max_turns:
            await self.sessions.update_status(session_id, "completed")
            raise ValueError("Maximum turns reached")
        
        # Increment turn
        turn_number = await self.sessions.increment_turn(session_id)
        
        # Create turn record
        turn = await self.turns.create(
            session_id=session_id,
            turn_number=turn_number,
            input_context=user_input,
        )
        
        # Get previous turn summary
        previous_summary = None
        if session.turns:
            last_turn = session.turns[-1]
            previous_summary = last_turn.coordinator_summary
        
        # Determine active agents
        config = SimulationConfig(**session.config)
        if focus_agents:
            active_roles = [r.value for r in focus_agents]
        else:
            active_roles = [r.value for r in config.agents]
        
        # Get recent messages for context
        recent_messages = []
        if len(session.turns) > 1:
            for prev_turn in session.turns[-3:]:
                recent_messages.extend(prev_turn.agent_responses or [])
        
        # Build context for agents
        context = AgentContext(
            session_id=session_id,
            turn_number=turn_number,
            scenario=session.scenario,
            max_turns=session.max_turns,
            active_agents=active_roles,
            previous_summary=previous_summary,
            recent_messages=recent_messages,
            user_input=user_input,
            custom_context=context_override or {},
        )
        
        # Execute agent turns
        agent_responses = await self._execute_agent_turns(
            context=context,
            active_roles=active_roles,
            config=config,
        )
        
        # Generate coordinator summary
        coordinator_summary = await self._generate_summary(agent_responses)
        
        # Calculate metrics
        duration_ms = int((time.monotonic() - start_time) * 1000)
        total_tokens = sum(
            r.token_usage.get("total_tokens", 0) for r in agent_responses
        )
        
        # Update turn record
        await self.turns.update(
            turn_id=turn.id,
            coordinator_summary=coordinator_summary,
            agent_responses=[r.to_dict() for r in agent_responses],
            duration_ms=duration_ms,
            token_usage={"total_tokens": total_tokens},
            status="completed",
        )
        
        # Store memories if enabled
        if config.enable_memory:
            await self._store_turn_memories(
                session_id=session_id,
                turn_number=turn_number,
                agent_responses=agent_responses,
            )
        
        await self.db.commit()
        
        # Check if simulation should end
        has_more_turns = turn_number < session.max_turns
        
        # Collect all tasks from agent responses
        all_tasks = []
        all_text = []
        for response in agent_responses:
            all_text.append(response.response)
            if response.task_distribution:
                for task in response.task_distribution:
                    task_with_id = {
                        "task_id": f"{response.agent_role}_{len(all_tasks)}",
                        "task_name": task.get("task_name", "Unnamed Task"),
                        "description": task.get("description", ""),
                        "estimated_effort_hours": task.get("estimated_effort_hours", 8.0),
                        "dependencies": task.get("dependencies", []),
                        "assigned_agent": response.agent_role,
                        "priority": task.get("priority", 5),
                    }
                    all_tasks.append(task_with_id)
        
        # Perform Function Point Analysis
        fp_estimation = None
        if all_tasks or all_text:
            combined_text = "\n".join(all_text)
            fp_estimation = self.fp_analyzer.analyze_text(combined_text, all_tasks)
        
        # Create Project Timeline
        project_timeline = None
        if all_tasks:
            # Get agent capacities (default 8 hours/day)
            agent_capacity = {
                role.value: 8.0 for role in config.agents
            }
            
            # Ensure created_at is a datetime object
            project_start = session.created_at
            if isinstance(project_start, str):
                try:
                    project_start = datetime.fromisoformat(project_start.replace('Z', '+00:00'))
                except (ValueError, AttributeError):
                    project_start = datetime.utcnow()
            elif not isinstance(project_start, datetime):
                project_start = datetime.utcnow()
            
            timeline_data = self.scheduler.create_timeline(
                tasks=all_tasks,
                project_start_date=project_start,
                agent_capacity=agent_capacity,
            )
            project_timeline = timeline_data
        
        logger.info(
            f"Executed turn {turn_number}",
            extra={
                "session_id": str(session_id),
                "duration_ms": duration_ms,
                "total_tokens": total_tokens,
            }
        )
        
        return {
            "session_id": session_id,
            "turn_number": turn_number,
            "status": TurnStatus.COMPLETED,
            "coordinator_summary": coordinator_summary,
            "agent_responses": [r.to_dict() for r in agent_responses],
            "duration_ms": duration_ms,
            "total_tokens": total_tokens,
            "has_more_turns": has_more_turns,
            "next_suggested_action": self._suggest_next_action(agent_responses),
            "project_timeline": project_timeline,
            "function_point_estimation": fp_estimation,
        }
    
    async def analyze_what_if(
        self,
        session_id: UUID,
        modifications: List[Dict[str, Any]],
        base_turn: Optional[int] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        num_turns: int = 3,
    ) -> Dict[str, Any]:
        """
        Perform what-if scenario analysis.
        
        Args:
            session_id: The session ID
            modifications: List of modifications to analyze
            base_turn: Turn to base analysis on (defaults to current)
            name: Optional scenario name
            description: Optional description
            num_turns: Number of turns to simulate
            
        Returns:
            What-if analysis results
        """
        # Get session
        session = await self.sessions.get_by_id(session_id, include_turns=True)
        
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        base_turn = base_turn or session.current_turn
        
        # Create what-if record
        what_if = await self.what_ifs.create(
            session_id=session_id,
            base_turn=base_turn,
            modifications=modifications,
            name=name,
            description=description,
        )
        
        # Get agent analyses
        config = SimulationConfig(**session.config)
        agent_analyses = []
        
        for role in config.agents:
            agent = AgentFactory.create(role.value)
            analysis = await agent.analyze_what_if(
                scenario=session.scenario,
                base_turn=base_turn,
                modifications=modifications,
                llm_client=self.llm,
            )
            agent_analyses.append(analysis)
        
        # Generate predicted outcomes
        predicted_outcomes = self._generate_predictions(
            agent_analyses=agent_analyses,
            num_turns=num_turns,
        )
        
        # Calculate confidence scores
        confidence_scores = {
            a["agent_role"]: a.get("confidence", 0.5)
            for a in agent_analyses
        }
        overall_confidence = sum(confidence_scores.values()) / len(confidence_scores)
        
        # Update what-if record
        await self.what_ifs.update_results(
            scenario_id=what_if.id,
            predicted_outcomes=predicted_outcomes,
            agent_analyses=agent_analyses,
            confidence_scores=confidence_scores,
            status="completed",
        )
        
        await self.db.commit()
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            agent_analyses=agent_analyses,
            predicted_outcomes=predicted_outcomes,
        )
        
        return {
            "id": what_if.id,
            "session_id": session_id,
            "base_turn": base_turn,
            "name": name,
            "modifications": modifications,
            "predicted_outcomes": predicted_outcomes,
            "agent_analyses": agent_analyses,
            "overall_confidence": overall_confidence,
            "recommendation": recommendation,
            "created_at": what_if.created_at,
        }
    
    async def get_session_state(self, session_id: UUID) -> Dict[str, Any]:
        """Get the current state of a simulation session."""
        session = await self.sessions.get_by_id(
            session_id,
            include_turns=True,
            include_agent_states=True,
        )
        
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        agent_states = await self.agent_states.get_by_session(session_id)
        
        return {
            "session_id": session.id,
            "name": session.name,
            "description": session.description,
            "scenario": session.scenario,
            "status": session.status,
            "current_turn": session.current_turn,
            "max_turns": session.max_turns,
            "config": session.config,
            "agent_states": [s.to_dict() for s in agent_states],
            "created_at": session.created_at,
            "updated_at": session.updated_at,
        }
    
    async def get_session_memory(self, session_id: UUID) -> Dict[str, Any]:
        """Get all memory for a session."""
        session = await self.sessions.get_by_id(session_id)
        
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        memories = await self.memories.get_by_session(session_id)
        
        # Group by agent
        memories_by_agent = {}
        for mem in memories:
            role = mem.agent_role
            if role not in memories_by_agent:
                memories_by_agent[role] = []
            memories_by_agent[role].append(mem.to_dict())
        
        return {
            "session_id": session_id,
            "total_memories": len(memories),
            "memories_by_agent": memories_by_agent,
        }
    
    async def _execute_agent_turns(
        self,
        context: AgentContext,
        active_roles: List[str],
        config: SimulationConfig,
    ) -> List[AgentResponse]:
        """Execute turns for all active agents."""
        responses = []
        
        # Create agents and execute in priority order
        agents = AgentFactory.create_all(roles=active_roles)
        
        for agent in agents:
            # Get relevant memories for this agent
            if config.enable_memory:
                memories = await self.memories.get_relevant(
                    session_id=context.session_id,
                    agent_role=agent.role,
                    limit=5,
                )
                context.memories = [m.to_dict() for m in memories]
            
            # Add previous responses to context
            context.recent_messages = [r.to_dict() for r in responses]
            
            # Generate response with agent-specific key assignment
            response = await agent.generate_response(
                context=context,
                llm_client=self.llm,
            )
            responses.append(response)
            
            # Update agent state
            await self.agent_states.increment_messages(
                session_id=context.session_id,
                agent_role=agent.role,
                turn_number=context.turn_number,
            )
        
        return responses
    
    async def _generate_summary(
        self,
        agent_responses: List[AgentResponse],
    ) -> str:
        """Generate coordinator summary of agent responses."""
        prompt = build_synthesis_prompt([r.to_dict() for r in agent_responses])
        
        result = await self.llm.complete(
            system_prompt="You are a simulation coordinator synthesizing team discussions.",
            user_prompt=prompt,
            temperature=0.5,
            max_tokens=500,
        )
        
        # Clean markdown from summary
        import re
        summary = result.content
        # Remove markdown formatting
        summary = re.sub(r'^#+\s*', '', summary, flags=re.MULTILINE)
        summary = re.sub(r'\*\*', '', summary)
        summary = re.sub(r'(?<!\*)\*(?!\*)', '', summary)
        summary = re.sub(r'__', '', summary)
        summary = re.sub(r'(?<!_)_(?!_)', '', summary)
        summary = re.sub(r'```[\s\S]*?```', '', summary)
        summary = re.sub(r'`', '', summary)
        summary = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', summary)
        summary = re.sub(r'//', '', summary)
        summary = re.sub(r'(?<!/)/(?!/)', '', summary)
        summary = re.sub(r'^[-*•]\s+', '', summary, flags=re.MULTILINE)
        summary = re.sub(r'^\d+\.\s+', '', summary, flags=re.MULTILINE)
        summary = re.sub(r'\n{3,}', '\n\n', summary)
        
        return summary.strip()
    
    async def _store_turn_memories(
        self,
        session_id: UUID,
        turn_number: int,
        agent_responses: List[AgentResponse],
    ) -> None:
        """Store memories from agent responses."""
        for response in agent_responses:
            # Store agent's contribution as memory
            await self.memories.create(
                session_id=session_id,
                agent_role=response.agent_role,
                memory_type=MemoryType.EPISODIC.value,
                key=f"turn_{turn_number}_response",
                content={
                    "turn": turn_number,
                    "summary": response.response[:500],
                    "actions": response.actions_proposed,
                },
                importance=0.6,
                expires_at=datetime.utcnow() + timedelta(hours=24),
            )
    
    def _suggest_next_action(
        self,
        agent_responses: List[AgentResponse],
    ) -> Optional[str]:
        """Suggest next action based on responses."""
        # Look for common action themes
        all_actions = []
        for r in agent_responses:
            all_actions.extend(r.actions_proposed)
        
        if all_actions:
            return f"Consider: {all_actions[0]}"
        return None
    
    def _generate_predictions(
        self,
        agent_analyses: List[Dict[str, Any]],
        num_turns: int,
    ) -> List[Dict[str, Any]]:
        """Generate predicted outcomes from agent analyses."""
        outcomes = []
        
        for turn in range(1, num_turns + 1):
            # Aggregate insights from agents
            key_changes = []
            risks = []
            opportunities = []
            
            for analysis in agent_analyses:
                key_changes.extend(analysis.get("recommended_actions", [])[:1])
                risks.extend(analysis.get("concerns", [])[:1])
                opportunities.extend(analysis.get("opportunities", [])[:1])
            
            avg_confidence = sum(
                a.get("confidence", 0.5) for a in agent_analyses
            ) / len(agent_analyses)
            
            outcomes.append({
                "turn": turn,
                "summary": f"Predicted state after {turn} turn(s) with modifications",
                "key_changes": key_changes[:3],
                "risk_factors": risks[:3],
                "opportunities": opportunities[:3],
                "probability": max(0.3, avg_confidence - (0.1 * turn)),
            })
        
        return outcomes
    
    def _generate_recommendation(
        self,
        agent_analyses: List[Dict[str, Any]],
        predicted_outcomes: List[Dict[str, Any]],
    ) -> str:
        """Generate overall recommendation."""
        # Simple heuristic-based recommendation
        total_confidence = sum(
            a.get("confidence", 0.5) for a in agent_analyses
        ) / len(agent_analyses)
        
        total_concerns = sum(
            len(a.get("concerns", [])) for a in agent_analyses
        )
        
        total_opportunities = sum(
            len(a.get("opportunities", [])) for a in agent_analyses
        )
        
        if total_confidence > 0.7 and total_opportunities > total_concerns:
            return "This scenario modification appears favorable. Consider proceeding with the changes."
        elif total_confidence < 0.4 or total_concerns > total_opportunities * 2:
            return "This scenario modification carries significant risks. Careful evaluation recommended."
        else:
            return "This scenario modification has mixed implications. Further analysis recommended."
