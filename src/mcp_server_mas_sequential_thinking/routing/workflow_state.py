"""Typed state models for Multi-Thinking workflow.

This module implements Agno 2.2.12's run_context pattern for type-safe
state management, replacing direct session_state dictionary manipulation.
"""

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MultiThinkingState:
    """Type-safe state for Multi-Thinking workflow execution.

    This replaces direct session_state dictionary manipulation with
    validated, type-safe state management using Agno 2.2.12 run_context API.

    Attributes:
        current_strategy: Current thinking strategy being executed
        current_complexity_score: Complexity score from AI analysis
        thinking_sequence: Sequence of thinking directions used
        cost_reduction: Estimated cost reduction percentage
        processing_stage: Current stage of workflow execution
        active_agents: List of currently executing agents
        completed_agents: List of successfully completed agents
        failed_agents: List of failed agents with errors
        start_time: Workflow start timestamp
        agent_timings: Execution time per agent (seconds)
        intermediate_results: Agent results for debugging
        token_usage: Token consumption per agent
    """

    # Core processing metadata
    current_strategy: str = "pending"
    current_complexity_score: float = 0.0
    thinking_sequence: list[str] = field(default_factory=list)
    cost_reduction: float = 0.0

    # Processing stage tracking
    processing_stage: str = (
        "initialization"  # initialization, analysis, synthesis, complete, error
    )

    # Agent execution tracking
    active_agents: list[str] = field(default_factory=list)
    completed_agents: list[str] = field(default_factory=list)
    failed_agents: list[str] = field(default_factory=list)

    # Performance metrics
    start_time: float | None = None
    agent_timings: dict[str, float] = field(default_factory=dict)

    # Intermediate results (for debugging and monitoring)
    intermediate_results: dict[str, str] = field(default_factory=dict)

    # Cost tracking
    token_usage: dict[str, dict[str, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize state with current timestamp if not provided."""
        if self.start_time is None:
            self.start_time = time.time()

    def mark_agent_started(self, agent_name: str) -> None:
        """Mark agent as started.

        Args:
            agent_name: Name of the agent that started
        """
        if agent_name not in self.active_agents:
            self.active_agents.append(agent_name)

    def mark_agent_completed(self, agent_name: str, result: str, timing: float) -> None:
        """Mark agent as completed with result.

        Args:
            agent_name: Name of the agent that completed
            result: Agent's output result
            timing: Execution time in seconds
        """
        if agent_name in self.active_agents:
            self.active_agents.remove(agent_name)
        if agent_name not in self.completed_agents:
            self.completed_agents.append(agent_name)

        # Store result and timing
        self.intermediate_results[agent_name] = result
        self.agent_timings[agent_name] = timing

    def mark_agent_failed(self, agent_name: str, error: str) -> None:
        """Mark agent as failed with error message.

        Args:
            agent_name: Name of the agent that failed
            error: Error message describing the failure
        """
        if agent_name in self.active_agents:
            self.active_agents.remove(agent_name)
        if agent_name not in self.failed_agents:
            self.failed_agents.append(agent_name)

        # Store error message
        self.intermediate_results[f"{agent_name}_error"] = error

    def record_token_usage(
        self, agent_name: str, input_tokens: int, output_tokens: int
    ) -> None:
        """Record token usage for an agent.

        Args:
            agent_name: Name of the agent
            input_tokens: Number of input tokens consumed
            output_tokens: Number of output tokens generated
        """
        self.token_usage[agent_name] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }

    @property
    def all_agents_complete(self) -> bool:
        """Check if all agents have completed execution.

        Returns:
            True if no agents are currently active, False otherwise
        """
        return len(self.active_agents) == 0

    @property
    def total_processing_time(self) -> float:
        """Calculate total processing time across all agents.

        Returns:
            Sum of all agent execution times in seconds
        """
        return sum(self.agent_timings.values())

    @property
    def elapsed_time(self) -> float:
        """Calculate elapsed time since workflow start.

        Returns:
            Time elapsed in seconds since workflow initialization
        """
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time

    @property
    def total_tokens_used(self) -> dict[str, int]:
        """Calculate total token usage across all agents.

        Returns:
            Dictionary with input_tokens, output_tokens, and total_tokens
        """
        total_input = sum(usage["input_tokens"] for usage in self.token_usage.values())
        total_output = sum(
            usage["output_tokens"] for usage in self.token_usage.values()
        )

        return {
            "input_tokens": total_input,
            "output_tokens": total_output,
            "total_tokens": total_input + total_output,
        }

    @property
    def success_rate(self) -> float:
        """Calculate agent success rate.

        Returns:
            Percentage of successfully completed agents (0.0 to 1.0)
        """
        total_executed = len(self.completed_agents) + len(self.failed_agents)
        if total_executed == 0:
            return 1.0
        return len(self.completed_agents) / total_executed

    def get_summary(self) -> dict[str, Any]:
        """Get comprehensive state summary for logging.

        Returns:
            Dictionary containing all key state metrics
        """
        return {
            "strategy": self.current_strategy,
            "complexity_score": self.current_complexity_score,
            "processing_stage": self.processing_stage,
            "active_agents": len(self.active_agents),
            "completed_agents": len(self.completed_agents),
            "failed_agents": len(self.failed_agents),
            "total_processing_time": self.total_processing_time,
            "elapsed_time": self.elapsed_time,
            "success_rate": self.success_rate,
            "total_tokens": self.total_tokens_used.get("total_tokens", 0),
        }

    def __repr__(self) -> str:
        """Return string representation of state."""
        return (
            f"MultiThinkingState("
            f"stage={self.processing_stage}, "
            f"strategy={self.current_strategy}, "
            f"active={len(self.active_agents)}, "
            f"completed={len(self.completed_agents)}, "
            f"failed={len(self.failed_agents)})"
        )
