"""
SQLAlchemy ORM models for Synthetic Workforce Simulator.

Defines the database schema for sessions, agents, turns, and memory storage.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

Base = declarative_base()


class TimestampMixin:
    """Mixin for created_at and updated_at timestamps."""
    
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )


class SimulationSession(Base, TimestampMixin):
    """
    Represents a simulation session.
    
    A session contains multiple turns where agents interact and collaborate.
    """
    
    __tablename__ = "simulation_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    
    # Session configuration
    scenario = Column(Text, nullable=False)
    config = Column(JSONB, nullable=False, default=dict)
    
    # Session state
    status = Column(
        String(50),
        nullable=False,
        default="active",
        index=True
    )  # active, paused, completed, failed
    current_turn = Column(Integer, nullable=False, default=0)
    max_turns = Column(Integer, nullable=False, default=50)
    
    # Metadata
    metadata_ = Column("metadata", JSONB, nullable=False, default=dict)
    
    # Relationships
    turns = relationship(
        "SimulationTurn",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="SimulationTurn.turn_number"
    )
    agent_states = relationship(
        "AgentState",
        back_populates="session",
        cascade="all, delete-orphan"
    )
    memories = relationship(
        "AgentMemory",
        back_populates="session",
        cascade="all, delete-orphan"
    )
    
    __table_args__ = (
        Index("ix_sessions_status_created", "status", "created_at"),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "scenario": self.scenario,
            "config": self.config,
            "status": self.status,
            "current_turn": self.current_turn,
            "max_turns": self.max_turns,
            "metadata": self.metadata_,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class AgentDefinition(Base, TimestampMixin):
    """
    Defines an agent role/type that can participate in simulations.
    
    This is the template for agent behavior, not an instance.
    """
    
    __tablename__ = "agent_definitions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    role = Column(String(100), nullable=False, unique=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    
    # Agent configuration
    system_prompt = Column(Text, nullable=False)
    capabilities = Column(JSONB, nullable=False, default=list)
    personality_traits = Column(JSONB, nullable=False, default=dict)
    
    # LLM configuration overrides
    temperature = Column(Float, nullable=True)
    max_tokens = Column(Integer, nullable=True)
    
    # Ordering and priority
    priority = Column(Integer, nullable=False, default=0)
    is_active = Column(Boolean, nullable=False, default=True)
    
    # Metadata
    metadata_ = Column("metadata", JSONB, nullable=False, default=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "role": self.role,
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities,
            "personality_traits": self.personality_traits,
            "priority": self.priority,
            "is_active": self.is_active,
        }


class AgentState(Base, TimestampMixin):
    """
    Represents the current state of an agent within a simulation session.
    """
    
    __tablename__ = "agent_states"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("simulation_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    agent_role = Column(String(100), nullable=False, index=True)
    
    # Agent state
    current_focus = Column(Text, nullable=True)
    goals = Column(JSONB, nullable=False, default=list)
    context = Column(JSONB, nullable=False, default=dict)
    
    # Interaction stats
    messages_sent = Column(Integer, nullable=False, default=0)
    last_active_turn = Column(Integer, nullable=True)
    
    # Relationships
    session = relationship("SimulationSession", back_populates="agent_states")
    
    __table_args__ = (
        UniqueConstraint("session_id", "agent_role", name="uq_agent_state_session_role"),
        Index("ix_agent_states_session_role", "session_id", "agent_role"),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "session_id": str(self.session_id),
            "agent_role": self.agent_role,
            "current_focus": self.current_focus,
            "goals": self.goals,
            "context": self.context,
            "messages_sent": self.messages_sent,
            "last_active_turn": self.last_active_turn,
        }


class SimulationTurn(Base, TimestampMixin):
    """
    Represents a single turn in a simulation.
    
    A turn contains all agent interactions for that round.
    """
    
    __tablename__ = "simulation_turns"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("simulation_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    turn_number = Column(Integer, nullable=False)
    
    # Turn context
    input_context = Column(Text, nullable=True)
    coordinator_summary = Column(Text, nullable=True)
    
    # Agent responses for this turn
    agent_responses = Column(JSONB, nullable=False, default=list)
    
    # Turn metrics
    duration_ms = Column(Integer, nullable=True)
    token_usage = Column(JSONB, nullable=False, default=dict)
    
    # Status
    status = Column(String(50), nullable=False, default="pending")
    error_message = Column(Text, nullable=True)
    
    # Relationships
    session = relationship("SimulationSession", back_populates="turns")
    
    __table_args__ = (
        UniqueConstraint("session_id", "turn_number", name="uq_turn_session_number"),
        Index("ix_turns_session_number", "session_id", "turn_number"),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "session_id": str(self.session_id),
            "turn_number": self.turn_number,
            "input_context": self.input_context,
            "coordinator_summary": self.coordinator_summary,
            "agent_responses": self.agent_responses,
            "duration_ms": self.duration_ms,
            "token_usage": self.token_usage,
            "status": self.status,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class AgentMemory(Base, TimestampMixin):
    """
    Stores agent memory/context that persists across turns.
    
    Memory can be short-term (within session) or long-term (across sessions).
    """
    
    __tablename__ = "agent_memories"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("simulation_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    agent_role = Column(String(100), nullable=False, index=True)
    
    # Memory content
    memory_type = Column(String(50), nullable=False, default="short_term")
    key = Column(String(255), nullable=False)
    content = Column(JSONB, nullable=False)
    
    # Memory metadata
    importance = Column(Float, nullable=False, default=0.5)
    access_count = Column(Integer, nullable=False, default=0)
    last_accessed_at = Column(DateTime(timezone=True), nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    session = relationship("SimulationSession", back_populates="memories")
    
    __table_args__ = (
        Index("ix_memories_session_agent_key", "session_id", "agent_role", "key"),
        Index("ix_memories_type_expires", "memory_type", "expires_at"),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "session_id": str(self.session_id),
            "agent_role": self.agent_role,
            "memory_type": self.memory_type,
            "key": self.key,
            "content": self.content,
            "importance": self.importance,
            "access_count": self.access_count,
            "last_accessed_at": self.last_accessed_at.isoformat() if self.last_accessed_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }


class WhatIfScenario(Base, TimestampMixin):
    """
    Stores what-if scenario analyses and their results.
    """
    
    __tablename__ = "what_if_scenarios"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("simulation_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    base_turn = Column(Integer, nullable=False)
    
    # Scenario definition
    name = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    modifications = Column(JSONB, nullable=False)
    
    # Results
    predicted_outcomes = Column(JSONB, nullable=True)
    agent_analyses = Column(JSONB, nullable=True)
    confidence_scores = Column(JSONB, nullable=True)
    
    # Status
    status = Column(String(50), nullable=False, default="pending")
    
    __table_args__ = (
        Index("ix_whatif_session_turn", "session_id", "base_turn"),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "session_id": str(self.session_id),
            "base_turn": self.base_turn,
            "name": self.name,
            "description": self.description,
            "modifications": self.modifications,
            "predicted_outcomes": self.predicted_outcomes,
            "agent_analyses": self.agent_analyses,
            "confidence_scores": self.confidence_scores,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
