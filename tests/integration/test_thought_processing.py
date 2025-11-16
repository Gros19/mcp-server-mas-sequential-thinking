"""Integration tests for thought processing workflow.

Tests the complete thought processing pipeline including validation,
routing, and response generation.
"""

from unittest.mock import MagicMock, patch

import pytest

from mcp_server_mas_sequential_thinking.core.models import ThoughtData
from mcp_server_mas_sequential_thinking.core.session import SessionMemory
from mcp_server_mas_sequential_thinking.services.server_core import (
    ThoughtProcessor,
)
from tests.fixtures.test_data import SAMPLE_API_KEYS, VALID_THOUGHTS


class TestThoughtProcessingIntegration:
    """Integration tests for the complete thought processing workflow."""

    def setup_method(self):
        """Set up test environment for each test."""
        self.session = SessionMemory()

    @pytest.mark.asyncio
    async def test_complete_thought_processing_workflow(self):
        """Test the complete workflow from thought input to response."""
        # Mock the workflow executor to avoid actual AI calls
        with patch(
            "mcp_server_mas_sequential_thinking.services.workflow_executor.WorkflowExecutor.execute_workflow"
        ) as mock_execute:
            # Set up mock response - return tuple (content, result, time)
            mock_result = MagicMock()
            mock_result.content = "This is a comprehensive analysis of the problem."
            mock_result.strategy_used = "multi_agent"
            mock_result.complexity_score = 45.0
            mock_result.step_name = "synthesis"
            mock_result.processing_time = 2.5

            # Mock returns tuple: (content, workflow_result, total_time)
            mock_execute.return_value = (
                "This is a comprehensive analysis of the problem.",
                mock_result,
                2.5,
            )

            # Create thought processor
            processor = ThoughtProcessor(self.session)

            # Create valid thought data
            sample_thought = VALID_THOUGHTS[0]
            thought_data = ThoughtData(
                thought=sample_thought.thought,
                thoughtNumber=sample_thought.thoughtNumber,
                totalThoughts=sample_thought.totalThoughts,
                nextThoughtNeeded=sample_thought.nextThoughtNeeded,
                isRevision=sample_thought.isRevision,
                branchFromThought=sample_thought.branchFromThought,
                branchId=sample_thought.branchId,
                needsMoreThoughts=sample_thought.needsMoreThoughts,
            )

            # Process the thought
            result = await processor.process_thought(thought_data)

            # Verify response
            assert isinstance(result, str)
            assert len(result) > 0
            assert "comprehensive analysis" in result.lower()

            # Verify workflow executor was called
            mock_execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_session_memory_integration(self):
        """Test that thoughts are properly stored in session memory."""
        with patch(
            "mcp_server_mas_sequential_thinking.services.workflow_executor.WorkflowExecutor.execute_workflow"
        ) as mock_execute:
            # Set up mock response - return tuple (content, result, time)
            mock_result = MagicMock()
            mock_result.content = "Analysis complete."
            mock_result.strategy_used = "single_agent"
            mock_result.complexity_score = 25.0
            mock_result.step_name = "analysis"
            mock_result.processing_time = 1.0

            # Mock returns tuple: (content, workflow_result, total_time)
            mock_execute.return_value = (
                "Analysis complete.",
                mock_result,
                1.0,
            )

            processor = ThoughtProcessor(self.session)

            # Process multiple thoughts
            for sample_thought in VALID_THOUGHTS:
                thought_data = ThoughtData(
                    thought=sample_thought.thought,
                    thoughtNumber=sample_thought.thoughtNumber,
                    totalThoughts=sample_thought.totalThoughts,
                    nextThoughtNeeded=sample_thought.nextThoughtNeeded,
                    isRevision=sample_thought.isRevision,
                    branchFromThought=sample_thought.branchFromThought,
                    branchId=sample_thought.branchId,
                    needsMoreThoughts=sample_thought.needsMoreThoughts,
                )

                await processor.process_thought(thought_data)

            # Verify thoughts are stored in session
            assert len(self.session.thought_history) == len(VALID_THOUGHTS)

            # Test thought retrieval
            for i, thought in enumerate(VALID_THOUGHTS, 1):
                stored_content = await self.session.find_thought_content(i)
                assert stored_content == thought.thought

    @pytest.mark.asyncio
    async def test_context_building_integration(self):
        """Test that context is properly built from session history."""
        with patch(
            "mcp_server_mas_sequential_thinking.services.workflow_executor.WorkflowExecutor.execute_workflow"
        ) as mock_execute:
            mock_result = MagicMock()
            mock_result.content = "Context-aware response."
            mock_result.strategy_used = "multi_agent"
            mock_result.complexity_score = 35.0
            mock_result.step_name = "synthesis"
            mock_result.processing_time = 2.0

            # Mock returns tuple: (content, workflow_result, total_time)
            mock_execute.return_value = (
                "Context-aware response.",
                mock_result,
                2.0,
            )

            processor = ThoughtProcessor(self.session)

            # Add some context to the session first
            context_thought = ThoughtData(
                thought="I need to understand the fundamentals first.",
                thoughtNumber=1,
                totalThoughts=2,
                nextThoughtNeeded=True,
                isRevision=False,
                branchFromThought=None,
                branchId=None,
                needsMoreThoughts=False,
            )

            await processor.process_thought(context_thought)

            # Now process a follow-up thought
            followup_thought = ThoughtData(
                thought="Building on the previous analysis, what are the next steps?",
                thoughtNumber=2,
                totalThoughts=2,
                nextThoughtNeeded=False,
                isRevision=False,
                branchFromThought=None,
                branchId=None,
                needsMoreThoughts=False,
            )

            await processor.process_thought(followup_thought)

            # Verify that execute_workflow was called with proper arguments
            assert mock_execute.call_count == 2  # Once for each thought

            # Get the second call's arguments
            second_call_args = mock_execute.call_args_list[1]
            thought_data_arg = second_call_args[0][0]
            input_prompt_arg = second_call_args[0][1]

            # Verify thought data is correct
            assert thought_data_arg.thoughtNumber == 2
            # Verify prompt contains the current thought content
            assert "Building on the previous analysis" in input_prompt_arg

    @pytest.mark.asyncio
    async def test_error_handling_integration(self):
        """Test error handling throughout the processing pipeline."""
        with patch(
            "mcp_server_mas_sequential_thinking.routing.MultiThinkingWorkflowRouter"
        ) as mock_router:
            # Set up router to raise an exception
            mock_router_instance = mock_router.return_value
            mock_router_instance.process_thought_workflow.side_effect = Exception(
                "AI service unavailable"
            )

            processor = ThoughtProcessor(self.session)

            thought_data = ThoughtData(
                thought="This will trigger an error.",
                thoughtNumber=1,
                totalThoughts=1,
                nextThoughtNeeded=False,
                isRevision=False,
                branchFromThought=None,
                branchId=None,
                needsMoreThoughts=False,
            )

            # Should raise ThoughtProcessingError
            from mcp_server_mas_sequential_thinking.core import ThoughtProcessingError

            with pytest.raises(ThoughtProcessingError):
                await processor.process_thought(thought_data)

    @pytest.mark.asyncio
    async def test_branching_workflow_integration(self):
        """Test branching workflow integration."""
        with patch(
            "mcp_server_mas_sequential_thinking.services.workflow_executor.WorkflowExecutor.execute_workflow"
        ) as mock_execute:
            mock_result = MagicMock()
            mock_result.content = "Branch analysis complete."
            mock_result.strategy_used = "multi_agent"
            mock_result.complexity_score = 50.0
            mock_result.step_name = "branch_synthesis"
            mock_result.processing_time = 3.0

            # Mock returns tuple: (content, workflow_result, total_time)
            mock_execute.return_value = (
                "Branch analysis complete.",
                mock_result,
                3.0,
            )

            processor = ThoughtProcessor(self.session)

            # Create main thought
            main_thought = ThoughtData(
                thought="What are the different approaches to this problem?",
                thoughtNumber=1,
                totalThoughts=1,
                nextThoughtNeeded=True,
                isRevision=False,
                branchFromThought=None,
                branchId=None,
                needsMoreThoughts=True,
            )

            await processor.process_thought(main_thought)

            # Create branch thought
            branch_thought = ThoughtData(
                thought="Approach A: Use machine learning algorithms.",
                thoughtNumber=2,
                totalThoughts=2,
                nextThoughtNeeded=False,
                isRevision=False,
                branchFromThought=1,
                branchId="approach_a",
                needsMoreThoughts=False,
            )

            result = await processor.process_thought(branch_thought)

            # Verify branching context
            assert len(self.session.thought_history) == 2
            assert result is not None
            assert "branch analysis" in result.lower()


class TestConfigurationIntegration:
    """Integration tests for configuration and environment setup."""

    @patch.dict(
        "os.environ",
        {
            "LLM_PROVIDER": "deepseek",
            "DEEPSEEK_API_KEY": SAMPLE_API_KEYS["valid"]["deepseek"],
        },
    )
    def test_valid_environment_configuration(self):
        """Test that valid environment configuration works end-to-end."""
        from mcp_server_mas_sequential_thinking.config import (
            validate_configuration_comprehensive,
        )

        result = validate_configuration_comprehensive()
        # Should not have any validation errors for required keys
        assert (
            "DEEPSEEK_API_KEY" not in result
            or "Required but not set" not in result.get("DEEPSEEK_API_KEY", "")
        )

    @patch.dict("os.environ", {}, clear=True)
    def test_missing_configuration_detection(self):
        """Test that missing configuration is properly detected."""
        from mcp_server_mas_sequential_thinking.config import (
            validate_configuration_comprehensive,
        )

        result = validate_configuration_comprehensive("deepseek")
        assert "DEEPSEEK_API_KEY" in result
        assert "Required but not set" in result["DEEPSEEK_API_KEY"]
