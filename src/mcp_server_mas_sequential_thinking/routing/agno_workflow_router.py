"""Multi-Thinking Workflow Router - Complete Rewrite.

Pure multi-directional thinking workflow implementation based on Agno v2.0.
Completely removes old complexity routing, focuses on multi-thinking.
"""

import logging
import re
import time
from dataclasses import dataclass
from typing import Any

from agno.workflow.step import Step
from agno.workflow.types import StepInput, StepOutput
from agno.workflow.workflow import Workflow

from mcp_server_mas_sequential_thinking.config.modernized_config import get_model_config

# Import at module level - moved from function to avoid PLC0415
from mcp_server_mas_sequential_thinking.core.models import ThoughtData
from mcp_server_mas_sequential_thinking.processors.multi_thinking_processor import (
    MultiThinkingSequentialProcessor,
    create_multi_thinking_step_output,
)
from mcp_server_mas_sequential_thinking.routing.ai_complexity_analyzer import (
    AIComplexityAnalyzer,
)
from mcp_server_mas_sequential_thinking.routing.workflow_state import (
    MultiThinkingState,
)

logger = logging.getLogger(__name__)


@dataclass
class MultiThinkingWorkflowResult:
    """Result from Multi-Thinking workflow execution."""

    content: str
    strategy_used: str
    processing_time: float
    complexity_score: float
    step_name: str
    thinking_sequence: list[str]
    cost_reduction: float


class MultiThinkingWorkflowRouter:
    """Pure Multi-Thinking workflow router using Agno v2.0."""

    def __init__(self) -> None:
        """Initialize Multi-Thinking workflow router."""
        self.model_config = get_model_config()
        self.complexity_analyzer = AIComplexityAnalyzer(self.model_config)

        # Initialize Multi-Thinking processor
        self.multi_thinking_processor = MultiThinkingSequentialProcessor()

        # Create workflow steps
        self.complexity_analysis_step = self._create_complexity_analysis_step()
        self.full_thinking_step = self._create_multi_thinking_step(
            step_name="full_sequence",
            forced_strategy_name="full_exploration",
        )

        # Create main workflow
        self.workflow = Workflow(
            name="multi_thinking_workflow",
            steps=[self.complexity_analysis_step, self.full_thinking_step],
        )

        logger.info("Multi-Thinking Workflow Router initialized")

    def _get_typed_state_from_session(
        self, session_state: dict[str, Any]
    ) -> MultiThinkingState:
        """Extract typed state from session_state dict.

        Provides type-safe access to workflow state while maintaining
        compatibility with Agno 2.2.12's session_state pattern.

        Args:
            session_state: Agno session_state dictionary

        Returns:
            Typed MultiThinkingState instance
        """
        return MultiThinkingState(
            current_strategy=session_state.get("current_strategy", "pending"),
            current_complexity_score=session_state.get("current_complexity_score", 0.0),
            thinking_sequence=session_state.get("thinking_sequence", []),
            cost_reduction=session_state.get("cost_reduction", 0.0),
            processing_stage=session_state.get("processing_stage", "initialization"),
            start_time=session_state.get("start_time"),
        )

    def _save_typed_state_to_session(
        self, state: MultiThinkingState, session_state: dict[str, Any]
    ) -> None:
        """Save typed state back to session_state dict.

        Converts typed state model to session_state dict format for
        compatibility with Agno 2.2.12.

        Args:
            state: Typed state to save
            session_state: Target session_state dictionary
        """
        session_state["current_strategy"] = state.current_strategy
        session_state["current_complexity_score"] = state.current_complexity_score
        session_state["thinking_sequence"] = state.thinking_sequence
        session_state["cost_reduction"] = state.cost_reduction
        session_state["processing_stage"] = state.processing_stage
        if state.start_time is not None:
            session_state["start_time"] = state.start_time

    def _create_complexity_analysis_step(self) -> Step:
        """Create AI complexity analysis step."""

        async def complexity_executor(
            step_input: StepInput, session_state: dict[str, Any]
        ) -> StepOutput:
            try:
                logger.info("🧠 COMPLEXITY ANALYSIS STEP:")

                if isinstance(step_input.input, dict):
                    thought_content = step_input.input.get(
                        "thought", str(step_input.input)
                    )
                    thought_number = step_input.input.get("thought_number", 1)
                    total_thoughts = step_input.input.get("total_thoughts", 1)
                    is_revision = step_input.input.get("isRevision", False)
                    branch_from_thought = step_input.input.get("branchFromThought")
                    branch_id = step_input.input.get("branchId")
                    needs_more_thoughts = step_input.input.get(
                        "needsMoreThoughts", False
                    )
                    next_thought_needed = step_input.input.get(
                        "nextThoughtNeeded", True
                    )
                else:
                    thought_content = str(step_input.input)
                    thought_number = 1
                    total_thoughts = 1
                    is_revision = False
                    branch_from_thought = None
                    branch_id = None
                    needs_more_thoughts = False
                    next_thought_needed = True

                thought_data = ThoughtData(
                    thought=thought_content,
                    thoughtNumber=thought_number,
                    totalThoughts=total_thoughts,
                    nextThoughtNeeded=next_thought_needed,
                    isRevision=is_revision,
                    branchFromThought=branch_from_thought,
                    branchId=branch_id,
                    needsMoreThoughts=needs_more_thoughts,
                )

                metrics = await self.complexity_analyzer.analyze(thought_data)
                session_state["current_complexity_score"] = metrics.complexity_score
                session_state["complexity_metrics"] = metrics
                session_state["complexity_score_scale"] = 100.0

                logger.info(
                    "  ✅ Complexity analyzed: %.1f",
                    metrics.complexity_score,
                )

                return StepOutput(
                    content="Complexity analysis completed.",
                    success=True,
                    step_name="complexity_analysis",
                )
            except Exception as exc:
                logger.exception("  ❌ Complexity analysis failed")
                return StepOutput(
                    content=f"Complexity analysis failed: {exc!s}",
                    success=False,
                    error=str(exc),
                    step_name="complexity_analysis_error",
                )

        executor: Any = complexity_executor
        return Step(
            name="complexity_analysis",
            executor=executor,
            description="AI complexity analysis for observability metadata",
        )

    def _create_multi_thinking_step(
        self, step_name: str, forced_strategy_name: str | None
    ) -> Step:
        """Create Six Thinking Hats processing step."""

        async def multi_thinking_executor(
            step_input: StepInput, session_state: dict[str, Any]
        ) -> StepOutput:
            """Execute Multi-Thinking thinking process with typed state.

            Uses typed state model for type-safety while maintaining Agno 2.2.12
            session_state compatibility.
            """
            try:
                logger.info("🎩 MULTI-THINKING STEP EXECUTION:")

                # Convert session_state to typed state for type-safe operations
                state = self._get_typed_state_from_session(session_state)
                state.processing_stage = "analysis"

                # Extract thought content and metadata
                if isinstance(step_input.input, dict):
                    thought_content = step_input.input.get(
                        "thought", str(step_input.input)
                    )
                    thought_number = step_input.input.get("thought_number", 1)
                    total_thoughts = step_input.input.get("total_thoughts", 1)
                    context = step_input.input.get("context", "")
                else:
                    thought_content = str(step_input.input)
                    thought_number = 1
                    total_thoughts = 1
                    context = ""

                # ThoughtData is now imported at module level

                # Extract context preservation fields from input if available
                if isinstance(step_input.input, dict):
                    is_revision = step_input.input.get("isRevision", False)
                    branch_from_thought = step_input.input.get("branchFromThought")
                    branch_id = step_input.input.get("branchId")
                    needs_more_thoughts = step_input.input.get(
                        "needsMoreThoughts", False
                    )
                    next_thought_needed = step_input.input.get(
                        "nextThoughtNeeded", True
                    )
                else:
                    is_revision = False
                    branch_from_thought = None
                    branch_id = None
                    needs_more_thoughts = False
                    next_thought_needed = True

                thought_data = ThoughtData(
                    thought=thought_content,
                    thoughtNumber=thought_number,
                    totalThoughts=total_thoughts,
                    nextThoughtNeeded=next_thought_needed,
                    isRevision=is_revision,
                    branchFromThought=branch_from_thought,
                    branchId=branch_id,
                    needsMoreThoughts=needs_more_thoughts,
                )

                logger.info("  📝 Input: %s...", thought_content[:100])
                logger.info("  🔢 Thought: %s/%s", thought_number, total_thoughts)

                complexity_metrics = session_state.get("complexity_metrics")

                # Process with Multi-Thinking
                result = (
                    await self.multi_thinking_processor.process_with_multi_thinking(
                        thought_data,
                        context,
                        state=state,
                        forced_strategy_name=forced_strategy_name,
                        complexity_metrics=complexity_metrics,
                    )
                )

                # Update typed state (type-safe operations)
                state.current_strategy = result.strategy_used
                state.current_complexity_score = result.complexity_score
                state.thinking_sequence = result.thinking_sequence
                state.cost_reduction = result.cost_reduction
                state.processing_stage = "synthesis_complete"

                # Save typed state back to session_state dict
                self._save_typed_state_to_session(state, session_state)

                logger.info("  ✅ Multi-Thinking completed: %s", result.strategy_used)
                logger.info("  📊 Complexity: %.1f", result.complexity_score)
                logger.info("  💰 Cost Reduction: %.1f%%", result.cost_reduction)
                logger.info("  📊 State: %s", state)

                return create_multi_thinking_step_output(result)

            except Exception as e:
                logger.exception("  ❌ Multi-Thinking execution failed")

                # Update state on error
                state = self._get_typed_state_from_session(session_state)
                state.processing_stage = "error"
                self._save_typed_state_to_session(state, session_state)

                return StepOutput(
                    content=f"Multi-Thinking processing failed: {e!s}",
                    success=False,
                    error=str(e),
                    step_name=f"{step_name}_error",
                )

        executor: Any = multi_thinking_executor
        return Step(
            name=step_name,
            executor=executor,
            description="Six Thinking Hats sequential processing",
        )

    async def process_thought_workflow(
        self, thought_data: "ThoughtData", context_prompt: str
    ) -> MultiThinkingWorkflowResult:
        """Process thought using Multi-Thinking workflow."""
        start_time = time.time()

        try:
            logger.info("🚀 MULTI-THINKING WORKFLOW INITIALIZATION:")
            logger.info(
                "  📝 Thought: %s%s",
                thought_data.thought[:100],
                "..." if len(thought_data.thought) > 100 else "",
            )
            logger.info(
                "  🔢 Thought Number: %s/%s",
                thought_data.thoughtNumber,
                thought_data.totalThoughts,
            )
            logger.info("  📋 Context Length: %d chars", len(context_prompt))
            logger.info(
                "  ⏰ Start Time: %s",
                time.strftime("%H:%M:%S", time.localtime(start_time)),
            )

            # Prepare workflow input for Multi-Thinking
            workflow_input = {
                "thought": thought_data.thought,
                "thought_number": thought_data.thoughtNumber,
                "total_thoughts": thought_data.totalThoughts,
                "context": context_prompt,
            }

            logger.info("📦 WORKFLOW INPUT PREPARATION:")
            logger.info("  📊 Input Keys: %s", list(workflow_input.keys()))
            logger.info("  📏 Input Size: %d chars", len(str(workflow_input)))

            # Initialize session_state for metadata tracking (with typed state bridge)
            session_state: dict[str, Any] = {
                "start_time": start_time,
                "thought_number": thought_data.thoughtNumber,
                "total_thoughts": thought_data.totalThoughts,
            }

            # Pre-initialize typed state
            initial_state = MultiThinkingState(start_time=start_time)
            self._save_typed_state_to_session(initial_state, session_state)

            logger.info("🎯 SESSION STATE SETUP (with typed state model):")
            logger.info("  📊 Initial State: %s", initial_state)
            logger.info("  🔢 Thought Number: %s", thought_data.thoughtNumber)
            logger.info("  📈 Total Thoughts: %s", thought_data.totalThoughts)

            logger.info(
                "▶️  EXECUTING Multi-Thinking workflow for thought #%s",
                thought_data.thoughtNumber,
            )

            # Execute Multi-Thinking workflow
            logger.info("🔄 WORKFLOW EXECUTION START...")
            result = await self.workflow.arun(
                input=workflow_input, session_state=session_state
            )
            logger.info("✅ WORKFLOW EXECUTION COMPLETED")

            processing_time = time.time() - start_time

            # Extract clean content from result
            content = self._extract_clean_content(result)

            logger.info("📋 CONTENT VALIDATION:")
            logger.info(
                "  ✅ Content extracted successfully: %d characters", len(content)
            )
            logger.info(
                "  📝 Content preview: %s%s",
                content[:150],
                "..." if len(content) > 150 else "",
            )

            # Get metadata from typed state (type-safe access)
            final_state = self._get_typed_state_from_session(session_state)
            complexity_score = final_state.current_complexity_score
            strategy_used = final_state.current_strategy
            thinking_sequence = final_state.thinking_sequence
            cost_reduction = final_state.cost_reduction

            logger.info("📊 WORKFLOW RESULT COMPILATION:")
            logger.info("  🎯 Strategy used: %s", strategy_used)
            logger.info("  🧠 Thinking sequence: %s", " → ".join(thinking_sequence))
            logger.info("  📈 Complexity score: %.1f", complexity_score)
            logger.info("  💰 Cost reduction: %.1f%%", cost_reduction)
            logger.info("  ⏱️  Processing time: %.3fs", processing_time)

            workflow_result = MultiThinkingWorkflowResult(
                content=content,
                strategy_used=strategy_used,
                processing_time=processing_time,
                complexity_score=complexity_score,
                step_name="multi_thinking_execution",
                thinking_sequence=thinking_sequence,
                cost_reduction=cost_reduction,
            )

            logger.info("🎉 MULTI-THINKING WORKFLOW COMPLETION:")
            logger.info(
                "  ✅ Completed: strategy=%s, time=%.3fs, score=%.1f, reduction=%.1f%%",
                strategy_used,
                processing_time,
                complexity_score,
                cost_reduction,
            )

            return workflow_result

        except Exception as e:
            processing_time = time.time() - start_time
            logger.exception(
                "Multi-Thinking workflow execution failed after %.3fs", processing_time
            )

            return MultiThinkingWorkflowResult(
                content=f"Error processing thought with Multi-Thinking: {e!s}",
                strategy_used="error_fallback",
                processing_time=processing_time,
                complexity_score=0.0,
                step_name="error_handling",
                thinking_sequence=[],
                cost_reduction=0.0,
            )

    def _extract_clean_content(self, result: Any) -> str:
        """Extract clean content from workflow result."""

        def extract_recursive(obj: Any, depth: int = 0) -> str:
            """Recursively extract clean content from nested objects."""
            if depth > 10:  # Prevent infinite recursion
                return str(obj)

            # Handle dictionary with common content keys
            if isinstance(obj, dict):
                for key in [
                    "result",
                    "content",
                    "message",
                    "text",
                    "response",
                    "output",
                    "answer",
                ]:
                    if key in obj:
                        return extract_recursive(obj[key], depth + 1)
                # Fallback to any string content
                for value in obj.values():
                    if isinstance(value, str) and len(value.strip()) > 10:
                        return value.strip()
                return str(obj)

            # Handle objects with content attributes
            if hasattr(obj, "content"):
                content = obj.content
                if isinstance(content, str):
                    return content.strip()
                return extract_recursive(content, depth + 1)

            # Handle other output objects
            if hasattr(obj, "output"):
                return extract_recursive(obj.output, depth + 1)

            # Handle list/tuple - extract first meaningful content
            if isinstance(obj, (list, tuple)) and obj:
                for item in obj:
                    result = extract_recursive(item, depth + 1)
                    if isinstance(result, str) and len(result.strip()) > 10:
                        return result.strip()

            # If it's already a string, clean it up
            if isinstance(obj, str):
                content = obj.strip()

                # Remove object representations
                if any(
                    content.startswith(pattern)
                    for pattern in [
                        "RunOutput(",
                        "TeamRunOutput(",
                        "StepOutput(",
                        "WorkflowResult(",
                        "{'result':",
                        '{"result":',
                        "{'content':",
                        '{"content":',
                    ]
                ):
                    # Try to extract content using regex (re imported at module level)

                    patterns = [
                        (r"content='([^']*)'", 1),
                        (r'content="([^"]*)"', 1),
                        (r"'result':\s*'([^']*)'", 1),
                        (r'"result":\s*"([^"]*)"', 1),
                        (r"'([^']{20,})'", 1),
                        (r'"([^"]{20,})"', 1),
                    ]

                    for pattern, group in patterns:
                        match = re.search(pattern, content)
                        if match:
                            extracted = match.group(group).strip()
                            if len(extracted) > 10:
                                return extracted

                    # Clean up object syntax
                    cleaned = re.sub(r'[{}()"\']', " ", content)
                    cleaned = re.sub(
                        r"\b(RunOutput|TeamRunOutput|StepOutput|content|result|success|error)\b",
                        " ",
                        cleaned,
                    )
                    cleaned = re.sub(r"\s+", " ", cleaned).strip()

                    if len(cleaned) > 20:
                        return cleaned

                return content

            # Fallback
            result = str(obj).strip()
            if len(result) > 20 and not any(
                result.startswith(pattern)
                for pattern in ["RunOutput(", "TeamRunOutput(", "StepOutput(", "<"]
            ):
                return result

            return "Multi-Thinking processing completed successfully"

        return extract_recursive(result)


# For backward compatibility with the old AgnoWorkflowRouter name
AgnoWorkflowRouter = MultiThinkingWorkflowRouter
WorkflowResult = MultiThinkingWorkflowResult
