"""
Pydantic schemas for request/response validation.

Defines all API schemas with comprehensive validation.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, ConfigDict


# =============================================================================
# Enums
# =============================================================================

class AgentRole(str, Enum):
    """Available agent roles in the simulation."""
    CEO = "ceo"
    PM = "pm"
    ENGINEERING_LEAD = "engineering_lead"
    DESIGNER = "designer"
    SALES = "sales"
    SUPPORT = "support"
    SIMULATION_ANALYST = "simulation_analyst"


class SessionStatus(str, Enum):
    """Simulation session status."""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class TurnStatus(str, Enum):
    """Simulation turn status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class MemoryType(str, Enum):
    """Types of agent memory."""
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


# =============================================================================
# Base Schemas
# =============================================================================

class BaseSchema(BaseModel):
    """Base schema with common configuration."""
    
    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )


class TimestampSchema(BaseSchema):
    """Schema with timestamp fields."""
    
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


# =============================================================================
# Agent Schemas
# =============================================================================

class AgentCapability(BaseSchema):
    """Agent capability definition."""
    
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., min_length=1, max_length=500)
    weight: float = Field(default=1.0, ge=0.0, le=1.0)


class AgentPersonality(BaseSchema):
    """Agent personality traits."""
    
    assertiveness: float = Field(default=0.5, ge=0.0, le=1.0)
    analytical: float = Field(default=0.5, ge=0.0, le=1.0)
    collaborative: float = Field(default=0.5, ge=0.0, le=1.0)
    creative: float = Field(default=0.5, ge=0.0, le=1.0)
    detail_oriented: float = Field(default=0.5, ge=0.0, le=1.0)


class AgentBase(BaseSchema):
    """Base agent schema."""
    
    role: AgentRole
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    capabilities: List[str] = Field(default_factory=list)
    personality_traits: Dict[str, float] = Field(default_factory=dict)


class AgentDefinitionResponse(AgentBase, TimestampSchema):
    """Agent definition response."""
    
    id: UUID
    priority: int
    is_active: bool


class AgentStateResponse(BaseSchema):
    """Agent state within a session."""
    
    id: UUID
    session_id: UUID
    agent_role: str
    current_focus: Optional[str] = None
    goals: List[str] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)
    messages_sent: int = 0
    last_active_turn: Optional[int] = None


class AgentListResponse(BaseSchema):
    """Response for listing agents."""
    
    agents: List[AgentDefinitionResponse]
    total: int


# =============================================================================
# Simulation Schemas
# =============================================================================

class SimulationConfig(BaseSchema):
    """Simulation configuration options."""
    
    max_turns: int = Field(default=50, ge=1, le=500)
    agents: List[AgentRole] = Field(
        default_factory=lambda: list(AgentRole)
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    enable_memory: bool = Field(default=True)
    enable_analytics: bool = Field(default=True)


class SimulationStartRequest(BaseSchema):
    """Request to start a new simulation."""
    
    scenario: str = Field(
        ...,
        min_length=10,
        max_length=5000,
        description="The simulation scenario description"
    )
    name: Optional[str] = Field(None, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    config: SimulationConfig = Field(default_factory=SimulationConfig)
    initial_context: Optional[Dict[str, Any]] = None
    
    @field_validator("scenario")
    @classmethod
    def validate_scenario(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Scenario cannot be empty or whitespace only")
        return v.strip()


class SimulationStartResponse(BaseSchema):
    """Response after starting a simulation."""
    
    session_id: UUID
    status: SessionStatus
    scenario: str
    config: SimulationConfig
    agents: List[str]
    message: str
    created_at: datetime


class SimulationNextRequest(BaseSchema):
    """Request to execute next turn."""
    
    session_id: UUID
    user_input: Optional[str] = Field(None, max_length=2000)
    focus_agents: Optional[List[AgentRole]] = None
    context_override: Optional[Dict[str, Any]] = None


class FunctionPoint(BaseSchema):
    """Function Point metric for estimation."""
    
    function_name: str
    function_type: str = Field(..., description="EI, EO, EQ, ILF, EIF")
    complexity: str = Field(default="average", description="low, average, high")
    fp_count: int = Field(ge=0)
    description: Optional[str] = None


class FunctionPointEstimation(BaseSchema):
    """Function Point based estimation."""
    
    total_function_points: int = Field(ge=0)
    unadjusted_fp: int = Field(ge=0)
    value_adjustment_factor: float = Field(default=1.0, ge=0.0, le=2.0)
    adjusted_fp: int = Field(ge=0)
    function_points: List[FunctionPoint] = Field(default_factory=list)
    estimated_hours: float = Field(ge=0.0)
    estimated_days: float = Field(ge=0.0)
    estimated_cost: Optional[float] = Field(None, ge=0.0)
    hourly_rate: Optional[float] = Field(None, ge=0.0, description="Default hourly rate for estimation")


class TaskSchedule(BaseSchema):
    """Task scheduling information."""
    
    task_id: str
    task_name: str
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    duration_days: float = Field(ge=0.0)
    assigned_agent: Optional[str] = None
    status: str = Field(default="planned", description="planned, in_progress, completed, blocked")
    dependencies: List[str] = Field(default_factory=list)
    effort_hours: float = Field(ge=0.0)
    completion_percentage: float = Field(default=0.0, ge=0.0, le=100.0)


class ProjectTimeline(BaseSchema):
    """Project timeline with Gantt chart data."""
    
    project_start_date: datetime
    project_end_date: Optional[datetime] = None
    total_duration_days: float = Field(ge=0.0)
    milestones: List[Dict[str, Any]] = Field(default_factory=list)
    tasks: List[TaskSchedule] = Field(default_factory=list)
    critical_path: List[str] = Field(default_factory=list)
    slack_days: Dict[str, float] = Field(default_factory=dict)


class TaskDistribution(BaseSchema):
    """Task distribution for an agent."""
    
    task_name: str
    description: str
    completion_chance: float = Field(ge=0.0, le=1.0)
    estimated_effort_hours: Optional[float] = None
    dependencies: List[str] = Field(default_factory=list)
    priority: int = Field(default=5, ge=1, le=10)
    function_points: Optional[FunctionPointEstimation] = None
    schedule: Optional[TaskSchedule] = None


class AgentMetrics(BaseSchema):
    """Metrics for an agent's performance."""
    
    productivity_score: float = Field(default=0.8, ge=0.0, le=1.0)
    collaboration_score: float = Field(default=0.8, ge=0.0, le=1.0)
    quality_score: float = Field(default=0.8, ge=0.0, le=1.0)
    engagement_level: float = Field(default=0.8, ge=0.0, le=1.0)
    tasks_completed: int = Field(default=0, ge=0)
    tasks_in_progress: int = Field(default=0, ge=0)
    tasks_blocked: int = Field(default=0, ge=0)
    response_time_ms: Optional[int] = None


class AgentTurnResponse(BaseSchema):
    """Response from a single agent for a turn."""
    
    agent_role: str
    response: str
    reasoning: Optional[str] = None
    actions_proposed: List[str] = Field(default_factory=list)
    directed_to: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    token_usage: Dict[str, int] = Field(default_factory=dict)
    metrics: Optional[AgentMetrics] = None
    task_distribution: List[TaskDistribution] = Field(default_factory=list)


class SimulationNextResponse(BaseSchema):
    """Response for simulation turn execution."""
    
    session_id: UUID
    turn_number: int
    status: TurnStatus
    coordinator_summary: str
    agent_responses: List[AgentTurnResponse]
    duration_ms: int
    total_tokens: int
    has_more_turns: bool
    next_suggested_action: Optional[str] = None
    project_timeline: Optional[ProjectTimeline] = None
    function_point_estimation: Optional[FunctionPointEstimation] = None


class SimulationStateResponse(BaseSchema):
    """Full simulation state response."""
    
    session_id: UUID
    name: Optional[str]
    description: Optional[str]
    scenario: str
    status: SessionStatus
    current_turn: int
    max_turns: int
    config: SimulationConfig
    agent_states: List[AgentStateResponse]
    created_at: datetime
    updated_at: datetime


class SimulationListItem(BaseSchema):
    """Simulation list item (simplified for listing)."""
    
    session_id: UUID
    name: Optional[str]
    description: Optional[str]
    scenario: str
    status: SessionStatus
    current_turn: int
    max_turns: int
    created_at: datetime
    updated_at: datetime


class SimulationListResponse(BaseSchema):
    """Response for listing simulations."""
    
    simulations: List[SimulationListItem]
    total: int


# =============================================================================
# What-If Schemas
# =============================================================================

class WhatIfModification(BaseSchema):
    """A single modification for what-if analysis."""
    
    type: str = Field(..., description="Type of modification: context, agent, scenario")
    target: str = Field(..., description="What to modify")
    change: Dict[str, Any] = Field(..., description="The change to apply")
    description: Optional[str] = None


class WhatIfRequest(BaseSchema):
    """Request for what-if scenario analysis."""
    
    session_id: UUID
    base_turn: Optional[int] = Field(
        None,
        description="Turn to base analysis on (defaults to current)"
    )
    name: Optional[str] = Field(None, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    modifications: List[WhatIfModification] = Field(
        ...,
        min_length=1,
        max_length=10
    )
    num_turns_to_simulate: int = Field(default=3, ge=1, le=10)


class WhatIfOutcome(BaseSchema):
    """Predicted outcome from what-if analysis."""
    
    turn: int
    summary: str
    key_changes: List[str]
    risk_factors: List[str]
    opportunities: List[str]
    probability: float = Field(ge=0.0, le=1.0)


class WhatIfAgentAnalysis(BaseSchema):
    """Agent-specific analysis for what-if scenario."""
    
    agent_role: str
    impact_assessment: str
    recommended_actions: List[str]
    concerns: List[str]
    confidence: float = Field(ge=0.0, le=1.0)


class WhatIfResponse(BaseSchema):
    """Response for what-if scenario analysis."""
    
    id: UUID
    session_id: UUID
    base_turn: int
    name: Optional[str]
    modifications: List[WhatIfModification]
    predicted_outcomes: List[WhatIfOutcome]
    agent_analyses: List[WhatIfAgentAnalysis]
    overall_confidence: float = Field(ge=0.0, le=1.0)
    recommendation: str
    created_at: datetime


# =============================================================================
# Memory Schemas
# =============================================================================

class MemoryEntry(BaseSchema):
    """A single memory entry."""
    
    id: UUID
    agent_role: str
    memory_type: MemoryType
    key: str
    content: Dict[str, Any]
    importance: float = Field(ge=0.0, le=1.0)
    access_count: int
    created_at: datetime
    last_accessed_at: Optional[datetime]
    expires_at: Optional[datetime]


class MemoryResponse(BaseSchema):
    """Response for memory retrieval."""
    
    session_id: UUID
    total_memories: int
    memories_by_agent: Dict[str, List[MemoryEntry]]
    summary: Optional[str] = None


class MemoryCreateRequest(BaseSchema):
    """Request to create a memory entry."""
    
    session_id: UUID
    agent_role: AgentRole
    memory_type: MemoryType = MemoryType.SHORT_TERM
    key: str = Field(..., min_length=1, max_length=255)
    content: Dict[str, Any]
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    ttl_hours: Optional[int] = Field(None, ge=1, le=720)


# =============================================================================
# Error Schemas
# =============================================================================

class ErrorDetail(BaseSchema):
    """Error detail information."""
    
    field: Optional[str] = None
    message: str
    code: Optional[str] = None


class ErrorResponse(BaseSchema):
    """Standard error response."""
    
    error: str
    message: str
    details: List[ErrorDetail] = Field(default_factory=list)
    request_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# Health Check Schemas
# =============================================================================

class ComponentHealth(BaseSchema):
    """Health status of a component."""
    
    name: str
    status: str  # healthy, degraded, unhealthy
    latency_ms: Optional[int] = None
    message: Optional[str] = None


class HealthResponse(BaseSchema):
    """Health check response."""
    
    status: str  # healthy, degraded, unhealthy
    version: str
    environment: str
    components: List[ComponentHealth]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# Pagination Schemas
# =============================================================================

class PaginationParams(BaseSchema):
    """Pagination parameters."""
    
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)
    
    @property
    def offset(self) -> int:
        return (self.page - 1) * self.page_size


class PaginatedResponse(BaseSchema):
    """Paginated response wrapper."""
    
    items: List[Any]
    total: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_prev: bool
