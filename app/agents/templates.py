"""
Prompt templates for agent interactions.

Provides structured prompts for different simulation contexts.
"""

from string import Template
from typing import Any, Dict, List, Optional


class PromptTemplate:
    """Template for generating prompts with variable substitution."""
    
    def __init__(self, template: str):
        self.template = Template(template)
    
    def render(self, **kwargs) -> str:
        """Render the template with provided variables."""
        return self.template.safe_substitute(**kwargs)


# =============================================================================
# Turn Context Templates
# =============================================================================

TURN_CONTEXT_TEMPLATE = PromptTemplate("""
## Simulation Context

**Scenario:** $scenario

**Current Turn:** $turn_number / $max_turns

**Previous Turn Summary:**
$previous_summary

**Your Role:** $agent_role ($agent_name)

**Active Agents This Turn:** $active_agents

---

## Recent Discussion

$recent_messages

---

## Your Task

Based on the scenario and discussion above, provide your perspective and contribution as the $agent_role.

$user_input

Remember to:
1. Stay in character as the $agent_role
2. Build on previous discussion points
3. Address relevant concerns raised by other agents
4. Propose concrete actions or insights
5. Keep your response focused and actionable
""")


FIRST_TURN_TEMPLATE = PromptTemplate("""
## New Simulation Started

**Scenario:** $scenario

**Your Role:** $agent_role ($agent_name)

**Team Members:** $active_agents

---

## Your Task

This is the first turn of the simulation. As the $agent_role, provide your initial perspective on the scenario.

Consider:
1. What are the key challenges and opportunities you see?
2. What questions would you want to explore?
3. What initial direction or priorities would you suggest?
4. What do you need from other team members?

Provide a focused, actionable response that sets the stage for productive discussion.
""")


# =============================================================================
# What-If Analysis Templates
# =============================================================================

WHAT_IF_ANALYSIS_TEMPLATE = PromptTemplate("""
## What-If Scenario Analysis

**Original Scenario:** $original_scenario

**Base Turn:** $base_turn

**Proposed Modifications:**
$modifications

---

## Your Analysis Task

As the $agent_role, analyze how these modifications would affect the simulation outcomes.

Consider:
1. How would this change affect your area of responsibility?
2. What new risks or opportunities would emerge?
3. What would you do differently given these changes?
4. What cascading effects might occur?

Provide a structured analysis with:
- Impact Assessment (High/Medium/Low)
- Key Changes You Would Make
- Risks to Watch
- Opportunities to Pursue
- Confidence Level (0-100%)
""")


# =============================================================================
# Memory Context Templates
# =============================================================================

MEMORY_CONTEXT_TEMPLATE = PromptTemplate("""
## Relevant Context from Previous Interactions

$memory_items

---

Use this context to inform your response, but focus on the current situation.
""")


# =============================================================================
# Coordinator Templates
# =============================================================================

COORDINATOR_SUMMARY_TEMPLATE = PromptTemplate("""
## Turn Summary

**Turn $turn_number Completed**

### Agent Contributions:

$agent_summaries

### Key Themes:
$key_themes

### Decisions Made:
$decisions

### Open Questions:
$open_questions

### Recommended Next Steps:
$next_steps
""")


COORDINATOR_SYNTHESIS_TEMPLATE = PromptTemplate("""
You are the simulation coordinator. Your job is to synthesize agent responses into a cohesive summary.

## Agent Responses This Turn:

$agent_responses

---

## Your Task:

Create a brief summary that includes:
1. **Key Themes** (2-3 main points of discussion)
2. **Areas of Agreement** (where agents aligned)
3. **Areas of Tension** (where agents disagreed)
4. **Decisions or Conclusions** (if any were reached)
5. **Open Questions** (what needs further discussion)
6. **Suggested Next Focus** (what should be discussed next)

Keep your summary concise and actionable (under 300 words).
""")


# =============================================================================
# Utility Functions
# =============================================================================

def format_agent_messages(messages: List[Dict[str, Any]]) -> str:
    """Format agent messages for inclusion in prompts."""
    if not messages:
        return "_No previous messages_"
    
    formatted = []
    for msg in messages:
        role = msg.get("agent_role", "Unknown")
        content = msg.get("response", msg.get("content", ""))
        formatted.append(f"**{role.upper()}:**\n{content}\n")
    
    return "\n".join(formatted)


def format_memory_items(memories: List[Dict[str, Any]]) -> str:
    """Format memory items for inclusion in prompts."""
    if not memories:
        return "_No relevant memories_"
    
    formatted = []
    for mem in memories:
        key = mem.get("key", "")
        content = mem.get("content", {})
        importance = mem.get("importance", 0.5)
        
        # Format content based on type
        if isinstance(content, dict):
            content_str = ", ".join(f"{k}: {v}" for k, v in content.items())
        else:
            content_str = str(content)
        
        formatted.append(f"- **{key}** (importance: {importance:.1f}): {content_str}")
    
    return "\n".join(formatted)


def format_modifications(modifications: List[Dict[str, Any]]) -> str:
    """Format what-if modifications for prompts."""
    if not modifications:
        return "_No modifications specified_"
    
    formatted = []
    for i, mod in enumerate(modifications, 1):
        mod_type = mod.get("type", "unknown")
        target = mod.get("target", "unknown")
        change = mod.get("change", {})
        desc = mod.get("description", "")
        
        formatted.append(f"{i}. **{mod_type.upper()}** on {target}")
        if desc:
            formatted.append(f"   Description: {desc}")
        formatted.append(f"   Change: {change}")
    
    return "\n".join(formatted)


def build_turn_prompt(
    scenario: str,
    turn_number: int,
    max_turns: int,
    agent_role: str,
    agent_name: str,
    active_agents: List[str],
    previous_summary: Optional[str] = None,
    recent_messages: Optional[List[Dict[str, Any]]] = None,
    memories: Optional[List[Dict[str, Any]]] = None,
    user_input: Optional[str] = None,
) -> str:
    """Build complete turn prompt for an agent."""
    
    # Format components
    agents_str = ", ".join(active_agents)
    messages_str = format_agent_messages(recent_messages or [])
    
    # Add memory context if available
    memory_context = ""
    if memories:
        memory_context = MEMORY_CONTEXT_TEMPLATE.render(
            memory_items=format_memory_items(memories)
        )
    
    # Format user input
    user_input_str = ""
    if user_input:
        user_input_str = f"\n**Additional Input:** {user_input}\n"
    
    # Choose template based on turn number
    if turn_number == 1:
        prompt = FIRST_TURN_TEMPLATE.render(
            scenario=scenario,
            agent_role=agent_role,
            agent_name=agent_name,
            active_agents=agents_str,
        )
    else:
        prompt = TURN_CONTEXT_TEMPLATE.render(
            scenario=scenario,
            turn_number=turn_number,
            max_turns=max_turns,
            agent_role=agent_role,
            agent_name=agent_name,
            active_agents=agents_str,
            previous_summary=previous_summary or "_No previous summary_",
            recent_messages=messages_str,
            user_input=user_input_str,
        )
    
    # Prepend memory context if available
    if memory_context:
        prompt = memory_context + "\n" + prompt
    
    return prompt


def build_what_if_prompt(
    original_scenario: str,
    base_turn: int,
    modifications: List[Dict[str, Any]],
    agent_role: str,
    agent_name: str,
) -> str:
    """Build what-if analysis prompt for an agent."""
    return WHAT_IF_ANALYSIS_TEMPLATE.render(
        original_scenario=original_scenario,
        base_turn=base_turn,
        modifications=format_modifications(modifications),
        agent_role=agent_role,
        agent_name=agent_name,
    )


def build_synthesis_prompt(agent_responses: List[Dict[str, Any]]) -> str:
    """Build coordinator synthesis prompt."""
    formatted_responses = []
    for resp in agent_responses:
        role = resp.get("agent_role", "Unknown")
        content = resp.get("response", "")
        formatted_responses.append(f"### {role.upper()}\n{content}\n")
    
    return COORDINATOR_SYNTHESIS_TEMPLATE.render(
        agent_responses="\n".join(formatted_responses)
    )
