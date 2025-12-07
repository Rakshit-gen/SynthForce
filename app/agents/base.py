"""
Base agent class and agent factory.

Provides the foundation for all agent types in the simulation.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import UUID

from app.agents.roles import get_agent_by_role, get_agent_priority
from app.agents.templates import build_turn_prompt, build_what_if_prompt

logger = logging.getLogger(__name__)


@dataclass
class AgentContext:
    """Context for an agent's current state."""
    
    session_id: UUID
    turn_number: int
    scenario: str
    max_turns: int
    active_agents: List[str]
    previous_summary: Optional[str] = None
    recent_messages: List[Dict[str, Any]] = field(default_factory=list)
    memories: List[Dict[str, Any]] = field(default_factory=list)
    user_input: Optional[str] = None
    custom_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResponse:
    """Response from an agent."""
    
    agent_role: str
    response: str
    reasoning: Optional[str] = None
    actions_proposed: List[str] = field(default_factory=list)
    directed_to: List[str] = field(default_factory=list)
    confidence: float = 0.8
    token_usage: Dict[str, int] = field(default_factory=dict)
    metrics: Optional[Dict[str, Any]] = None
    task_distribution: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "agent_role": self.agent_role,
            "response": self.response,
            "reasoning": self.reasoning,
            "actions_proposed": self.actions_proposed,
            "directed_to": self.directed_to,
            "confidence": self.confidence,
            "token_usage": self.token_usage,
        }
        if self.metrics:
            result["metrics"] = self.metrics
        if self.task_distribution:
            result["task_distribution"] = self.task_distribution
        return result


class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    
    Provides common functionality and interface for agent implementations.
    """
    
    def __init__(
        self,
        role: str,
        name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        self.role = role
        
        # Load defaults from role definition
        role_config = get_agent_by_role(role)
        self.name = name or role_config["name"]
        self.system_prompt = system_prompt or role_config["system_prompt"]
        self.capabilities = role_config.get("capabilities", [])
        self.personality = role_config.get("personality_traits", {})
        self.priority = role_config.get("priority", 0)
        
        # LLM configuration
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # State
        self.current_focus: Optional[str] = None
        self.goals: List[str] = []
        self.context: Dict[str, Any] = {}
    
    @property
    def identifier(self) -> str:
        """Unique identifier for the agent."""
        return f"{self.role}:{self.name}"
    
    def build_prompt(self, context: AgentContext) -> str:
        """Build the prompt for this agent given the context."""
        return build_turn_prompt(
            scenario=context.scenario,
            turn_number=context.turn_number,
            max_turns=context.max_turns,
            agent_role=self.role,
            agent_name=self.name,
            active_agents=context.active_agents,
            previous_summary=context.previous_summary,
            recent_messages=context.recent_messages,
            memories=context.memories,
            user_input=context.user_input,
        )
    
    def build_what_if_prompt(
        self,
        scenario: str,
        base_turn: int,
        modifications: List[Dict[str, Any]],
    ) -> str:
        """Build the what-if analysis prompt for this agent."""
        return build_what_if_prompt(
            original_scenario=scenario,
            base_turn=base_turn,
            modifications=modifications,
            agent_role=self.role,
            agent_name=self.name,
        )
    
    @abstractmethod
    async def generate_response(
        self,
        context: AgentContext,
        llm_client: Any,
    ) -> AgentResponse:
        """
        Generate a response for the current turn.
        
        Args:
            context: The current simulation context
            llm_client: The LLM client for generation
            
        Returns:
            AgentResponse with the agent's contribution
        """
        pass
    
    @abstractmethod
    async def analyze_what_if(
        self,
        scenario: str,
        base_turn: int,
        modifications: List[Dict[str, Any]],
        llm_client: Any,
    ) -> Dict[str, Any]:
        """
        Analyze a what-if scenario.
        
        Args:
            scenario: The original scenario
            base_turn: The turn to base analysis on
            modifications: List of proposed modifications
            llm_client: The LLM client for generation
            
        Returns:
            Analysis results
        """
        pass
    
    def update_state(
        self,
        current_focus: Optional[str] = None,
        goals: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update the agent's internal state."""
        if current_focus is not None:
            self.current_focus = current_focus
        if goals is not None:
            self.goals = goals
        if context is not None:
            self.context.update(context)
    
    def get_state(self) -> Dict[str, Any]:
        """Get the agent's current state."""
        return {
            "role": self.role,
            "name": self.name,
            "current_focus": self.current_focus,
            "goals": self.goals,
            "context": self.context,
            "capabilities": self.capabilities,
            "personality": self.personality,
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(role={self.role}, name={self.name})"


class SimulationAgent(BaseAgent):
    """
    Concrete agent implementation for simulation.
    
    Uses the Groq LLM client to generate responses.
    """
    
    async def generate_response(
        self,
        context: AgentContext,
        llm_client: Any,
    ) -> AgentResponse:
        """Generate a response using the LLM."""
        prompt = self.build_prompt(context)
        
        try:
            # Generate response from LLM with agent-specific key assignment
            result = await llm_client.complete(
                system_prompt=self.system_prompt,
                user_prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                agent_role=self.role,  # Pass agent role for consistent key assignment
            )
            
            # Parse and structure the response
            response = AgentResponse(
                agent_role=self.role,
                response=result.content,
                token_usage={
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "total_tokens": result.total_tokens,
                },
            )
            
            # Extract additional metadata from response
            response = self._enrich_response(response, result.content)
            
            logger.debug(
                f"Agent {self.role} generated response",
                extra={
                    "turn": context.turn_number,
                    "tokens": result.total_tokens,
                }
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Agent {self.role} failed to generate response: {e}")
            return AgentResponse(
                agent_role=self.role,
                response=f"[Error generating response: {str(e)}]",
                confidence=0.0,
            )
    
    async def analyze_what_if(
        self,
        scenario: str,
        base_turn: int,
        modifications: List[Dict[str, Any]],
        llm_client: Any,
    ) -> Dict[str, Any]:
        """Analyze what-if scenario using LLM."""
        prompt = self.build_what_if_prompt(scenario, base_turn, modifications)
        
        try:
            result = await llm_client.complete(
                system_prompt=self.system_prompt,
                user_prompt=prompt,
                temperature=0.5,  # Lower temperature for analysis
                max_tokens=self.max_tokens,
                agent_role=self.role,  # Pass agent role for consistent key assignment
            )
            
            # Parse the analysis response
            analysis = self._parse_what_if_analysis(result.content)
            analysis["agent_role"] = self.role
            
            return analysis
            
        except Exception as e:
            logger.error(f"Agent {self.role} failed what-if analysis: {e}")
            return {
                "agent_role": self.role,
                "error": str(e),
                "confidence": 0.0,
            }
    
    def _clean_markdown(self, text: str) -> str:
        """Remove markdown formatting from text."""
        import re
        
        cleaned = text
        # Remove markdown headers
        cleaned = re.sub(r'^#+\s*', '', cleaned, flags=re.MULTILINE)
        # Remove bold/italic markers
        cleaned = re.sub(r'\*\*', '', cleaned)
        cleaned = re.sub(r'(?<!\*)\*(?!\*)', '', cleaned)  # Remove single asterisks but not double
        cleaned = re.sub(r'__', '', cleaned)
        cleaned = re.sub(r'(?<!_)_(?!_)', '', cleaned)  # Remove single underscores but not double
        # Remove code blocks
        cleaned = re.sub(r'```[\s\S]*?```', '', cleaned)
        cleaned = re.sub(r'`', '', cleaned)
        # Remove links but keep text
        cleaned = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', cleaned)
        # Remove slashes used for emphasis
        cleaned = re.sub(r'//', '', cleaned)
        cleaned = re.sub(r'(?<!/)/(?!/)', '', cleaned)  # Remove single slashes but not double
        # Remove bullet markers but keep content
        cleaned = re.sub(r'^[-*•]\s+', '', cleaned, flags=re.MULTILINE)
        # Remove numbered list markers but keep content
        cleaned = re.sub(r'^\d+\.\s+', '', cleaned, flags=re.MULTILINE)
        # Clean up extra whitespace
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        return cleaned.strip()
    
    def _enrich_response(self, response: AgentResponse, content: str) -> AgentResponse:
        """Extract additional metadata from the response content."""
        import re
        import random
        
        # Clean markdown from response text
        response.response = self._clean_markdown(response.response)
        
        # Look for action items (lines starting with action keywords)
        action_keywords = ["ACTION:", "PROPOSE:", "RECOMMEND:", "TODO:", "-", "*"]
        actions = []
        
        for line in content.split("\n"):
            line_upper = line.strip().upper()
            for keyword in action_keywords:
                if line_upper.startswith(keyword) or (line_upper.startswith("-") and len(line.strip()) > 10):
                    # Clean markdown formatting from action text
                    action_text = self._clean_markdown(line.strip())
                    if action_text and len(action_text) > 5:
                        actions.append(action_text)
                    break
        
        if actions:
            response.actions_proposed = actions[:5]  # Limit to 5 actions
        
        # Look for directed messages (@ mentions)
        mentions = re.findall(r"@(\w+)", content.lower())
        if mentions:
            response.directed_to = list(set(mentions))  # Remove duplicates
        
        # Calculate metrics based on response characteristics
        response.metrics = self._calculate_metrics(response, content)
        
        # Generate task distribution from actions
        response.task_distribution = self._generate_task_distribution(actions, response.confidence)
        
        # Clean task names and descriptions
        for task in response.task_distribution:
            if "task_name" in task:
                task["task_name"] = self._clean_markdown(task["task_name"])
            if "description" in task:
                task["description"] = self._clean_markdown(task["description"])
        
        return response
    
    def _calculate_metrics(self, response: AgentResponse, content: str) -> Dict[str, Any]:
        """Calculate agent metrics based on response."""
        import random
        
        # Base metrics influenced by confidence and response quality
        base_score = response.confidence
        
        # Productivity: based on number of actions and response length
        action_count = len(response.actions_proposed)
        productivity = min(0.95, base_score + (action_count * 0.1))
        
        # Collaboration: based on mentions and directed messages
        collaboration = min(0.95, base_score + (len(response.directed_to) * 0.15))
        
        # Quality: based on response length and structure
        word_count = len(content.split())
        quality = min(0.95, base_score + min(word_count / 500, 0.2))
        
        # Engagement: based on confidence and response detail
        engagement = min(0.95, base_score + (word_count / 1000))
        
        # Task metrics
        tasks_completed = random.randint(0, min(3, action_count))
        tasks_in_progress = max(0, action_count - tasks_completed)
        tasks_blocked = random.randint(0, 1) if action_count > 2 else 0
        
        return {
            "productivity_score": round(productivity, 2),
            "collaboration_score": round(collaboration, 2),
            "quality_score": round(quality, 2),
            "engagement_level": round(engagement, 2),
            "tasks_completed": tasks_completed,
            "tasks_in_progress": tasks_in_progress,
            "tasks_blocked": tasks_blocked,
            "response_time_ms": random.randint(500, 3000),
        }
    
    def _generate_task_distribution(self, actions: List[str], confidence: float) -> List[Dict[str, Any]]:
        """Generate task distribution from actions."""
        import random
        
        tasks = []
        task_types = {
            "design": ["Design", "Create", "Prototype", "Wireframe"],
            "development": ["Implement", "Build", "Develop", "Code", "Integrate"],
            "planning": ["Plan", "Define", "Analyze", "Research", "Evaluate"],
            "testing": ["Test", "Validate", "Verify", "QA"],
            "documentation": ["Document", "Write", "Specify"],
        }
        
        for i, action in enumerate(actions[:5]):  # Limit to 5 tasks
            # Determine task type
            task_type = "planning"
            for key, keywords in task_types.items():
                if any(keyword.lower() in action.lower() for keyword in keywords):
                    task_type = key
                    break
            
            # Calculate completion chance based on confidence and task complexity
            base_chance = confidence
            complexity_modifier = random.uniform(-0.2, 0.1)
            completion_chance = max(0.3, min(0.95, base_chance + complexity_modifier))
            
            # Estimate effort (in hours)
            effort_hours = random.uniform(2, 16)
            
            # Priority (1-10, higher is more important)
            priority = random.randint(6, 10) if i < 2 else random.randint(4, 7)
            
            # Generate task ID
            task_id = f"task_{self.role}_{i}_{hash(action) % 10000}"
            
            tasks.append({
                "task_id": task_id,
                "task_name": action[:50],  # Truncate if too long
                "description": action,
                "completion_chance": round(completion_chance, 2),
                "estimated_effort_hours": round(effort_hours, 1),
                "dependencies": [],
                "priority": priority,
            })
        
        return tasks
    
    def _parse_what_if_analysis(self, content: str) -> Dict[str, Any]:
        """Parse what-if analysis from LLM response."""
        analysis = {
            "impact_assessment": "medium",
            "recommended_actions": [],
            "concerns": [],
            "opportunities": [],
            "confidence": 0.7,
            "raw_analysis": content,
        }
        
        # Simple parsing - look for key sections
        content_lower = content.lower()
        
        if "high" in content_lower and "impact" in content_lower:
            analysis["impact_assessment"] = "high"
        elif "low" in content_lower and "impact" in content_lower:
            analysis["impact_assessment"] = "low"
        
        # Extract confidence if mentioned
        import re
        confidence_match = re.search(r"confidence[:\s]+(\d+)%?", content_lower)
        if confidence_match:
            analysis["confidence"] = int(confidence_match.group(1)) / 100.0
        
        return analysis


class AgentFactory:
    """Factory for creating agent instances."""
    
    _agent_cache: Dict[str, BaseAgent] = {}
    
    @classmethod
    def create(
        cls,
        role: str,
        name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        use_cache: bool = True,
    ) -> BaseAgent:
        """
        Create or retrieve an agent instance.
        
        Args:
            role: The agent role
            name: Optional custom name
            system_prompt: Optional custom system prompt
            temperature: Optional custom temperature
            max_tokens: Optional custom max tokens
            use_cache: Whether to use cached instances
            
        Returns:
            BaseAgent instance
        """
        cache_key = f"{role}:{name or 'default'}"
        
        if use_cache and cache_key in cls._agent_cache:
            return cls._agent_cache[cache_key]
        
        agent = SimulationAgent(
            role=role,
            name=name,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        if use_cache:
            cls._agent_cache[cache_key] = agent
        
        return agent
    
    @classmethod
    def create_all(cls, roles: Optional[List[str]] = None) -> List[BaseAgent]:
        """
        Create all agents for a simulation.
        
        Args:
            roles: Optional list of roles to create (defaults to all)
            
        Returns:
            List of agent instances in priority order
        """
        from app.agents.roles import ALL_ROLES
        
        roles_to_create = roles or ALL_ROLES
        agents = [cls.create(role) for role in roles_to_create]
        
        # Sort by priority
        agents.sort(key=lambda a: a.priority, reverse=True)
        
        return agents
    
    @classmethod
    def clear_cache(cls) -> None:
        """Clear the agent cache."""
        cls._agent_cache.clear()
