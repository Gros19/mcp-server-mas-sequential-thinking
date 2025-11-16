"""Unit tests for Multi-Thinking workflow state management.

Tests the typed state model introduced with Agno 2.2.12 run_context pattern.
"""

import time

import pytest

from mcp_server_mas_sequential_thinking.routing.workflow_state import (
    MultiThinkingState,
)


class TestMultiThinkingState:
    """Test suite for MultiThinkingState dataclass."""

    def test_state_initialization_defaults(self):
        """Test that state initializes with correct defaults."""
        state = MultiThinkingState()

        assert state.current_strategy == "pending"
        assert state.current_complexity_score == 0.0
        assert state.thinking_sequence == []
        assert state.cost_reduction == 0.0
        assert state.processing_stage == "initialization"
        assert state.active_agents == []
        assert state.completed_agents == []
        assert state.failed_agents == []
        assert state.start_time is not None  # Should auto-initialize
        assert state.agent_timings == {}
        assert state.intermediate_results == {}
        assert state.token_usage == {}

    def test_state_initialization_with_start_time(self):
        """Test state initialization with explicit start time."""
        specific_time = time.time()
        state = MultiThinkingState(start_time=specific_time)

        assert state.start_time == specific_time

    def test_mark_agent_started(self):
        """Test marking agent as started."""
        state = MultiThinkingState()

        state.mark_agent_started("factual")
        assert "factual" in state.active_agents
        assert len(state.active_agents) == 1

        # Should not duplicate
        state.mark_agent_started("factual")
        assert len(state.active_agents) == 1

    def test_mark_agent_completed(self):
        """Test marking agent as completed with results."""
        state = MultiThinkingState()

        state.mark_agent_started("factual")
        state.mark_agent_completed("factual", "Test result", 1.5)

        assert "factual" not in state.active_agents
        assert "factual" in state.completed_agents
        assert state.intermediate_results["factual"] == "Test result"
        assert state.agent_timings["factual"] == 1.5

    def test_mark_agent_failed(self):
        """Test marking agent as failed with error."""
        state = MultiThinkingState()

        state.mark_agent_started("emotional")
        state.mark_agent_failed("emotional", "API timeout")

        assert "emotional" not in state.active_agents
        assert "emotional" in state.failed_agents
        assert state.intermediate_results["emotional_error"] == "API timeout"

    def test_record_token_usage(self):
        """Test recording token usage for agents."""
        state = MultiThinkingState()

        state.record_token_usage("factual", 100, 50)

        assert "factual" in state.token_usage
        assert state.token_usage["factual"]["input_tokens"] == 100
        assert state.token_usage["factual"]["output_tokens"] == 50
        assert state.token_usage["factual"]["total_tokens"] == 150

    def test_all_agents_complete_property(self):
        """Test all_agents_complete property."""
        state = MultiThinkingState()

        # Initially true (no active agents)
        assert state.all_agents_complete is True

        state.mark_agent_started("factual")
        assert state.all_agents_complete is False

        state.mark_agent_completed("factual", "Done", 1.0)
        assert state.all_agents_complete is True

    def test_total_processing_time_property(self):
        """Test total_processing_time calculation."""
        state = MultiThinkingState()

        state.mark_agent_started("factual")
        state.mark_agent_completed("factual", "Result 1", 1.5)

        state.mark_agent_started("emotional")
        state.mark_agent_completed("emotional", "Result 2", 2.3)

        assert state.total_processing_time == pytest.approx(3.8)

    def test_elapsed_time_property(self):
        """Test elapsed_time calculation."""
        start = time.time()
        state = MultiThinkingState(start_time=start)

        time.sleep(0.1)  # Sleep 100ms

        elapsed = state.elapsed_time
        assert elapsed >= 0.1
        assert elapsed < 1.0  # Should be less than 1 second

    def test_total_tokens_used_property(self):
        """Test total token usage calculation."""
        state = MultiThinkingState()

        state.record_token_usage("factual", 100, 50)
        state.record_token_usage("emotional", 200, 100)

        total = state.total_tokens_used

        assert total["input_tokens"] == 300
        assert total["output_tokens"] == 150
        assert total["total_tokens"] == 450

    def test_success_rate_property(self):
        """Test success rate calculation."""
        state = MultiThinkingState()

        # No executions yet - should be 100%
        assert state.success_rate == 1.0

        state.mark_agent_completed("factual", "Done", 1.0)
        assert state.success_rate == 1.0

        state.mark_agent_failed("emotional", "Error")
        assert state.success_rate == 0.5  # 1 success, 1 failure

        state.mark_agent_completed("critical", "Done", 1.0)
        assert state.success_rate == pytest.approx(0.6667, rel=1e-4)  # 2 success, 1 failure

    def test_get_summary(self):
        """Test state summary generation."""
        state = MultiThinkingState()
        state.current_strategy = "full_sequence"
        state.current_complexity_score = 8.5
        state.processing_stage = "synthesis"

        state.mark_agent_started("factual")
        state.mark_agent_completed("factual", "Result", 1.5)
        state.record_token_usage("factual", 100, 50)

        summary = state.get_summary()

        assert summary["strategy"] == "full_sequence"
        assert summary["complexity_score"] == 8.5
        assert summary["processing_stage"] == "synthesis"
        assert summary["active_agents"] == 0
        assert summary["completed_agents"] == 1
        assert summary["failed_agents"] == 0
        assert summary["total_processing_time"] == 1.5
        assert summary["total_tokens"] == 150
        assert "success_rate" in summary
        assert "elapsed_time" in summary

    def test_repr(self):
        """Test string representation."""
        state = MultiThinkingState()
        state.current_strategy = "double_hat"
        state.processing_stage = "analysis"
        state.mark_agent_started("factual")

        repr_str = repr(state)

        assert "MultiThinkingState" in repr_str
        assert "stage=analysis" in repr_str
        assert "strategy=double_hat" in repr_str
        assert "active=1" in repr_str
        assert "completed=0" in repr_str
        assert "failed=0" in repr_str

    def test_complex_workflow_scenario(self):
        """Test a complete workflow scenario."""
        state = MultiThinkingState()
        state.current_strategy = "full_sequence"
        state.current_complexity_score = 9.2
        state.thinking_sequence = ["factual", "emotional", "critical", "optimistic", "creative", "synthesis"]
        state.cost_reduction = 15.5
        state.processing_stage = "execution"

        # Simulate multi-agent execution
        agents = ["factual", "emotional", "critical", "optimistic", "creative"]

        for agent in agents:
            state.mark_agent_started(agent)

        # Complete some agents
        state.mark_agent_completed("factual", "Factual analysis complete", 1.2)
        state.record_token_usage("factual", 150, 80)

        state.mark_agent_completed("emotional", "Emotional perspective done", 0.9)
        state.record_token_usage("emotional", 100, 60)

        # Fail one agent
        state.mark_agent_failed("critical", "Model timeout")

        # Complete remaining
        state.mark_agent_completed("optimistic", "Opportunities identified", 1.1)
        state.record_token_usage("optimistic", 120, 70)

        state.mark_agent_completed("creative", "Creative solutions found", 1.5)
        state.record_token_usage("creative", 180, 100)

        state.processing_stage = "synthesis"

        # Verify final state
        assert len(state.completed_agents) == 4
        assert len(state.failed_agents) == 1
        assert state.success_rate == 0.8  # 4 out of 5 succeeded
        assert state.all_agents_complete is True
        assert state.total_processing_time == pytest.approx(4.7)  # 1.2 + 0.9 + 1.1 + 1.5

        total_tokens = state.total_tokens_used
        assert total_tokens["input_tokens"] == 550  # 150 + 100 + 120 + 180
        assert total_tokens["output_tokens"] == 310  # 80 + 60 + 70 + 100
        assert total_tokens["total_tokens"] == 860

        summary = state.get_summary()
        assert summary["strategy"] == "full_sequence"
        assert summary["processing_stage"] == "synthesis"
        assert summary["success_rate"] == 0.8
