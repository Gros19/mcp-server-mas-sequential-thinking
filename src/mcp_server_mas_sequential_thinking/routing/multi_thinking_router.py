"""Multi-Thinking Intelligent Router.

Intelligent routing system based on question complexity and type, supporting:
- Single direction mode: Fast processing for simple questions
- Double sequence: Balanced processing for moderate questions
- Triple core: Deep processing for standard questions
- Full multi-directional: Comprehensive processing for complex questions
"""

# Lazy import to break circular dependency
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .ai_complexity_analyzer import AIComplexityAnalyzer
from .complexity_types import ComplexityMetrics

if TYPE_CHECKING:
    from mcp_server_mas_sequential_thinking.core.models import ThoughtData

logger = logging.getLogger(__name__)
from mcp_server_mas_sequential_thinking.processors.multi_thinking_core import (
    ProcessingDepth,
    ThinkingDirection,
)


@dataclass
class ThinkingSequenceStrategy:
    """Thinking sequence strategy."""

    name: str
    complexity: ProcessingDepth
    thinking_sequence: list[ThinkingDirection]
    estimated_time_seconds: int
    description: str


class ThinkingSequenceLibrary:
    """Thinking sequence strategy library."""

    # Predefined thinking sequence strategies
    STRATEGIES = {
        # Single direction mode strategies
        "single_factual": ThinkingSequenceStrategy(
            name="Single Factual Mode",
            complexity=ProcessingDepth.SINGLE,
            thinking_sequence=[ThinkingDirection.FACTUAL],
            estimated_time_seconds=120,
            description="Pure fact collection, fast information processing",
        ),
        "single_intuitive": ThinkingSequenceStrategy(
            name="Single Intuitive Mode",
            complexity=ProcessingDepth.SINGLE,
            thinking_sequence=[ThinkingDirection.EMOTIONAL],
            estimated_time_seconds=30,
            description="Quick intuitive response, 30-second emotional judgment",
        ),
        "single_creative": ThinkingSequenceStrategy(
            name="Single Creative Mode",
            complexity=ProcessingDepth.SINGLE,
            thinking_sequence=[ThinkingDirection.CREATIVE],
            estimated_time_seconds=240,
            description="Creative generation mode, free innovative thinking",
        ),
        "single_critical": ThinkingSequenceStrategy(
            name="Single Critical Mode",
            complexity=ProcessingDepth.SINGLE,
            thinking_sequence=[ThinkingDirection.CRITICAL],
            estimated_time_seconds=120,
            description="Risk identification, fast critical analysis",
        ),
        # Double sequence strategies
        "evaluate_idea": ThinkingSequenceStrategy(
            name="Idea Evaluation Sequence",
            complexity=ProcessingDepth.DOUBLE,
            thinking_sequence=[
                ThinkingDirection.OPTIMISTIC,
                ThinkingDirection.CRITICAL,
            ],
            estimated_time_seconds=240,
            description="First examine benefits, then risks, balanced evaluation",
        ),
        "improve_design": ThinkingSequenceStrategy(
            name="Design Improvement Sequence",
            complexity=ProcessingDepth.DOUBLE,
            thinking_sequence=[ThinkingDirection.CRITICAL, ThinkingDirection.CREATIVE],
            estimated_time_seconds=360,
            description="Identify problems, then innovate improvements",
        ),
        "fact_and_judge": ThinkingSequenceStrategy(
            name="Fact and Judgment Sequence",
            complexity=ProcessingDepth.TRIPLE,
            thinking_sequence=[
                ThinkingDirection.FACTUAL,
                ThinkingDirection.CRITICAL,
                ThinkingDirection.SYNTHESIS,
            ],
            estimated_time_seconds=360,
            description="Collect facts, critical verification, comprehensive synthesis",
        ),
        # Triple core sequence strategies
        "problem_solving": ThinkingSequenceStrategy(
            name="Problem Solving Sequence",
            complexity=ProcessingDepth.TRIPLE,
            thinking_sequence=[
                ThinkingDirection.FACTUAL,
                ThinkingDirection.CREATIVE,
                ThinkingDirection.CRITICAL,
            ],
            estimated_time_seconds=480,
            description="Facts→Creativity→Evaluation, standard problem solving",
        ),
        "decision_making": ThinkingSequenceStrategy(
            name="Decision Making Sequence",
            complexity=ProcessingDepth.TRIPLE,
            thinking_sequence=[
                ThinkingDirection.EMOTIONAL,
                ThinkingDirection.OPTIMISTIC,
                ThinkingDirection.CRITICAL,
            ],
            estimated_time_seconds=390,
            description="Intuition→Value→Risk, rapid decision making",
        ),
        "philosophical_thinking": ThinkingSequenceStrategy(
            name="Philosophical Thinking Sequence",
            complexity=ProcessingDepth.TRIPLE,
            thinking_sequence=[
                ThinkingDirection.FACTUAL,
                ThinkingDirection.CREATIVE,
                ThinkingDirection.SYNTHESIS,
            ],
            estimated_time_seconds=540,
            description=(
                "Facts→Creation→Integration, deep philosophical thinking "
                "(resolves synthesis+review separation issue)"
            ),
        ),
        # Full multi-directional sequence
        "full_exploration": ThinkingSequenceStrategy(
            name="Full Exploration Sequence",
            complexity=ProcessingDepth.FULL,
            thinking_sequence=[
                ThinkingDirection.SYNTHESIS,
                ThinkingDirection.FACTUAL,
                ThinkingDirection.EMOTIONAL,
                ThinkingDirection.OPTIMISTIC,
                ThinkingDirection.CRITICAL,
                ThinkingDirection.CREATIVE,
                ThinkingDirection.SYNTHESIS,
            ],
            estimated_time_seconds=780,
            description=(
                "Complete multi-directional sequence, comprehensive deep analysis"
            ),
        ),
        "creative_innovation": ThinkingSequenceStrategy(
            name="Creative Innovation Sequence",
            complexity=ProcessingDepth.FULL,
            thinking_sequence=[
                ThinkingDirection.SYNTHESIS,
                ThinkingDirection.EMOTIONAL,
                ThinkingDirection.CREATIVE,
                ThinkingDirection.FACTUAL,
                ThinkingDirection.OPTIMISTIC,
                ThinkingDirection.CRITICAL,
                ThinkingDirection.SYNTHESIS,
            ],
            estimated_time_seconds=840,
            description="Innovation-prioritized complete workflow",
        ),
    }

    @classmethod
    def get_strategy(cls, strategy_name: str) -> ThinkingSequenceStrategy | None:
        """Get specified strategy."""
        return cls.STRATEGIES.get(strategy_name)

    @classmethod
    def get_strategies_by_complexity(
        cls, complexity: ProcessingDepth
    ) -> list[ThinkingSequenceStrategy]:
        """Get strategies by complexity level."""
        return [
            strategy
            for strategy in cls.STRATEGIES.values()
            if strategy.complexity == complexity
        ]


@dataclass
class RoutingDecision:
    """Routing decision result."""

    strategy: ThinkingSequenceStrategy
    reasoning: str
    complexity_metrics: ComplexityMetrics
    estimated_cost_reduction: float  # Cost reduction vs original system
    problem_type: str  # AI-determined problem type
    thinking_modes_needed: list[str]  # AI-recommended thinking modes


class MultiThinkingIntelligentRouter:
    """Multi-directional thinking intelligent router."""

    def __init__(self, complexity_analyzer: AIComplexityAnalyzer | None = None) -> None:
        self.complexity_analyzer = complexity_analyzer or AIComplexityAnalyzer()
        self.sequence_library = ThinkingSequenceLibrary()

        # Complexity threshold configuration
        self.complexity_thresholds = {
            ProcessingDepth.SINGLE: (0, 3),
            ProcessingDepth.DOUBLE: (3, 10),
            ProcessingDepth.TRIPLE: (10, 20),
            ProcessingDepth.FULL: (20, 100),
        }

    async def route_thought(self, thought_data: "ThoughtData") -> RoutingDecision:
        """AI-powered intelligent routing to optimal thinking sequence."""
        logger.info("AI-driven multi-thinking routing started")
        if logger.isEnabledFor(logging.INFO):
            logger.info("Input preview: %s", thought_data.thought[:100])

        # Step 1: AI analysis (complexity + problem type + thinking modes)
        complexity_metrics = await self.complexity_analyzer.analyze(thought_data)
        complexity_score = complexity_metrics.complexity_score

        # Extract AI analysis results directly from ComplexityMetrics
        problem_type = complexity_metrics.primary_problem_type
        thinking_modes_needed = complexity_metrics.thinking_modes_needed or [
            "SYNTHESIS"
        ]

        logger.info(
            "AI analysis - Complexity: %.1f, Type: %s, Modes: %s",
            complexity_score,
            problem_type,
            thinking_modes_needed,
        )

        # Step 2: Determine complexity level
        complexity_level = self._determine_complexity_level(complexity_score)
        logger.info("Complexity level determined: %s", complexity_level.value)

        # Step 3: AI-driven strategy selection
        strategy = self._select_optimal_strategy(
            complexity_level, problem_type, thinking_modes_needed, complexity_score
        )

        # Step 4: Generate reasoning
        reasoning = self._generate_reasoning(
            strategy, problem_type, thinking_modes_needed, complexity_metrics
        )

        # Step 5: Estimate cost reduction
        cost_reduction = self._estimate_cost_reduction(strategy, complexity_score)

        decision = RoutingDecision(
            strategy=strategy,
            reasoning=reasoning,
            complexity_metrics=complexity_metrics,
            estimated_cost_reduction=cost_reduction,
            problem_type=problem_type,
            thinking_modes_needed=thinking_modes_needed,
        )

        logger.info("Strategy selected: %s", strategy.name)
        if logger.isEnabledFor(logging.INFO):
            sequence = [direction.value for direction in strategy.thinking_sequence]
            logger.info("Thinking sequence: %s", sequence)
        logger.info("Estimated cost reduction: %.1f%%", cost_reduction)

        return decision

    def _determine_complexity_level(self, score: float) -> ProcessingDepth:
        """Determine processing level based on complexity score."""
        for level, (min_score, max_score) in self.complexity_thresholds.items():
            if min_score <= score < max_score:
                return level
        return ProcessingDepth.FULL

    def _select_optimal_strategy(
        self,
        complexity_level: ProcessingDepth,
        problem_type: str,
        thinking_modes_needed: list[str],
        complexity_score: float,
    ) -> ThinkingSequenceStrategy:
        """AI-driven strategy selection."""
        # Get strategies by complexity level
        candidate_strategies = self.sequence_library.get_strategies_by_complexity(
            complexity_level
        )

        if not candidate_strategies:
            logger.warning(
                f"No strategies found for complexity {complexity_level}, using fallback"
            )
            return self._get_fallback_strategy(complexity_level)

        # AI-driven selection based on problem type and thinking modes
        return self._select_by_ai_analysis(
            candidate_strategies, problem_type, thinking_modes_needed, complexity_score
        )

    def _select_by_ai_analysis(
        self,
        strategies: list[ThinkingSequenceStrategy],
        problem_type: str,
        thinking_modes_needed: list[str],
        complexity_score: float,
    ) -> ThinkingSequenceStrategy:
        """AI-driven strategy selection logic."""
        # Single mode AI-driven selection
        if strategies[0].complexity == ProcessingDepth.SINGLE:
            if "FACTUAL" in thinking_modes_needed:
                return self.sequence_library.get_strategy("single_factual")
            if "CREATIVE" in thinking_modes_needed:
                return self.sequence_library.get_strategy("single_creative")
            if "EMOTIONAL" in thinking_modes_needed:
                return self.sequence_library.get_strategy("single_intuitive")
            if "CRITICAL" in thinking_modes_needed:
                return self.sequence_library.get_strategy("single_critical")
            return self.sequence_library.get_strategy("single_factual")  # Default

        # For other complexity levels: intelligent selection based on problem type
        if problem_type == "PHILOSOPHICAL":
            return self.sequence_library.get_strategy("philosophical_thinking")
        if problem_type == "DECISION":
            return self.sequence_library.get_strategy("decision_making")
        if problem_type == "CREATIVE":
            return (
                self.sequence_library.get_strategy("creative_innovation")
                or strategies[0]
            )
        if problem_type == "EVALUATIVE":
            return self.sequence_library.get_strategy("evaluate_idea") or strategies[0]

        # Default: return first strategy
        return strategies[0]

    def _get_fallback_strategy(
        self, complexity_level: ProcessingDepth
    ) -> ThinkingSequenceStrategy:
        """Get fallback strategy."""
        fallback_map = {
            ProcessingDepth.SINGLE: "single_factual",
            ProcessingDepth.DOUBLE: "fact_and_judge",
            ProcessingDepth.TRIPLE: "problem_solving",
            ProcessingDepth.FULL: "full_exploration",
        }
        strategy_name = fallback_map.get(complexity_level, "single_factual")
        return self.sequence_library.get_strategy(strategy_name)

    def _generate_reasoning(
        self,
        strategy: ThinkingSequenceStrategy,
        problem_type: str,
        thinking_modes_needed: list[str],
        metrics: ComplexityMetrics,
    ) -> str:
        """Generate AI-driven routing decision reasoning."""
        reasoning_parts = [
            f"Strategy: {strategy.name}",
            f"AI Problem Type: {problem_type}",
            f"Complexity: {metrics.complexity_score:.1f}/100",
            (
                f"Thinking Sequence: "
                f"{' → '.join(direction.value for direction in strategy.thinking_sequence)}"
            ),
            f"Estimated Time: {strategy.estimated_time_seconds}s",
            f"AI Recommended Modes: {', '.join(thinking_modes_needed)}",
        ]

        # Add AI insights
        if "PHILOSOPHICAL" in problem_type:
            reasoning_parts.append("Deep philosophical analysis required")
        if "CREATIVE" in thinking_modes_needed:
            reasoning_parts.append("Creative thinking essential")
        if "SYNTHESIS" in thinking_modes_needed:
            reasoning_parts.append("Integration and synthesis needed")

        return " | ".join(reasoning_parts)

    def _estimate_cost_reduction(
        self, strategy: ThinkingSequenceStrategy, complexity_score: float
    ) -> float:
        """Estimate cost reduction compared to original system."""
        # Original system cost estimation (based on complexity)
        if complexity_score < 5:
            original_cost = 100  # Single agent cost baseline
        elif complexity_score < 15:
            original_cost = 300  # Mixed team cost
        else:
            original_cost = 600  # Full multi-agent cost

        # New system cost (based on thinking direction count and time)
        thinking_count = len(strategy.thinking_sequence)
        new_cost = thinking_count * 50 + strategy.estimated_time_seconds * 0.1

        # Calculate reduction percentage
        if original_cost > 0:
            reduction = max(0, (original_cost - new_cost) / original_cost * 100)
        else:
            reduction = 0

        return min(reduction, 85)  # Maximum 85% reduction


# Convenience functions
def create_multi_thinking_router(
    complexity_analyzer=None,
) -> MultiThinkingIntelligentRouter:
    """Create multi-directional thinking intelligent router."""
    return MultiThinkingIntelligentRouter(complexity_analyzer)


async def route_thought_to_thinking(thought_data: "ThoughtData") -> RoutingDecision:
    """Route thought to optimal thinking sequence."""
    router = MultiThinkingIntelligentRouter()
    return await router.route_thought(thought_data)
