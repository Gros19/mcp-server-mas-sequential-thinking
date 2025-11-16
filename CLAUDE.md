# CLAUDE.md

## Essential Commands

```bash
# Setup & Development
uv pip install -e ".[dev]"                              # Install dependencies
uv run mcp-server-mas-sequential-thinking               # Run server
uv run ruff check . --fix && uv run ruff format . && uv run mypy .  # Code quality

# Testing Framework
python run_tests.py                                     # Run all tests with coverage
python run_tests.py --unit --security                   # Run unit and security tests
uv run pytest tests/ -v                                 # Direct pytest execution

# Debugging & Monitoring
tail -f ~/.sequential_thinking/logs/sequential_thinking.log          # Live logs
npx @modelcontextprotocol/inspector uv run python src/mcp_server_mas_sequential_thinking/main.py  # Test server
```

## Project Overview

**AI-powered Multi-Thinking implementation** using Agno v2.2.12 framework via MCP. Processes thoughts through six cognitive perspectives (Factual, Emotional, Critical, Optimistic, Creative, Synthesis) with intelligent complexity analysis determining execution strategy (Single/Double/Triple/Full sequences).

**Core Flow:** External LLM → `sequentialthinking` tool → AI complexity analysis → Multi-Thinking workflow → Individual hat agents → Synthesis

**Recent Upgrade:** Migrated to Agno 2.2.12 with typed state management and message history optimization (Nov 2025)

## Configuration

**Required Environment Variables:**
```bash
LLM_PROVIDER=deepseek                                    # Provider (deepseek, groq, openrouter, ollama, github, anthropic)
DEEPSEEK_API_KEY=your_key                               # Provider API key
DEEPSEEK_ENHANCED_MODEL_ID=deepseek-chat                # Synthesis model
DEEPSEEK_STANDARD_MODEL_ID=deepseek-chat                # Individual hats model
EXA_API_KEY=your_key                                    # Optional: Research capabilities
```

**Model Strategy:**
- **Enhanced Model**: Blue Hat (synthesis) for complex integration
- **Standard Model**: Individual hats (White, Red, Black, Yellow, Green) for focused thinking
- **AI Selection**: System automatically chooses model based on hat type and complexity

## Key Architecture

**Entry Point:** `src/mcp_server_mas_sequential_thinking/main.py`

**Core Services:**
- `ThoughtProcessor`: Main orchestrator with dependency injection
- `WorkflowExecutor`: Manages Multi-Thinking workflow execution
- `AIComplexityAnalyzer`: AI-driven complexity assessment (replaces rule-based patterns)
- `MultiThinkingSequentialProcessor`: Executes chosen thinking sequence

**Processing Strategies (AI-Determined):**
1. **Single Hat**: Simple focused thinking
2. **Double Hat**: Two-step sequences (e.g., Optimistic→Critical)
3. **Triple Hat**: Core philosophical thinking (Factual→Creative→Synthesis)
4. **Full Sequence**: Complete Multi-Thinking with Blue Hat orchestration

## Critical Development Patterns

**Dependency Injection:** Manual constructor injection, Protocol-based interfaces in `core/types.py`

**Import Safety:** Avoid circular dependencies:
```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from module import Class
```

**Thread Safety:** Global state uses async locks:
```python
_processor_lock = asyncio.Lock()
async with _processor_lock:
    # Safe initialization
```

**Error Handling:** Use `ThoughtProcessingError` hierarchy, include `ProcessingMetadata` for debugging

**Parallel Processing:** Non-synthesis agents use `asyncio.gather` for simultaneous execution

---

## Agno 2.2.12 Modern Patterns (Nov 2025)

### Typed State Management

**Pattern:** Use typed state models instead of raw session_state dictionaries for type safety.

**Implementation:** `routing/workflow_state.py`

```python
from mcp_server_mas_sequential_thinking.routing.workflow_state import MultiThinkingState

# Get typed state from session_state
state = self._get_typed_state_from_session(session_state)

# Type-safe operations with IDE support
state.current_strategy = "full_sequence"
state.current_complexity_score = 8.5
state.processing_stage = "synthesis"

# Track agent execution
state.mark_agent_started("factual")
state.mark_agent_completed("factual", result, timing=1.2)
state.record_token_usage("factual", input_tokens=150, output_tokens=80)

# Save back to session_state
self._save_typed_state_to_session(state, session_state)

# Get comprehensive summary
summary = state.get_summary()
# {strategy: "full_sequence", completed_agents: 5, total_tokens: 850, ...}
```

**Benefits:**
- ✅ Type safety with IDE autocomplete
- ✅ Runtime validation
- ✅ No silent failures from typos
- ✅ Rich state tracking (agents, tokens, timing)
- ✅ Easy debugging with state.get_summary()

### Message History Optimization

**Pattern:** Control context window per agent to reduce token usage by 40-60%.

**Configuration:** `processors/multi_thinking_processor.py`

```python
MESSAGE_HISTORY_CONFIG = {
    ThinkingDirection.FACTUAL: 5,       # Recent context for data gathering
    ThinkingDirection.EMOTIONAL: 0,     # Fresh perspective without bias
    ThinkingDirection.CRITICAL: 3,      # Focused risk analysis
    ThinkingDirection.OPTIMISTIC: 3,    # Focused opportunity analysis
    ThinkingDirection.CREATIVE: 8,      # Broader context for creativity
    ThinkingDirection.SYNTHESIS: 10,    # Maximum context for integration
}

# Usage in agent execution
history_limit = MESSAGE_HISTORY_CONFIG.get(thinking_direction, 5)
result = await agent.arun(
    input=thought_data.thought,
    num_history_messages=history_limit  # Agno 2.2.12+ parameter
)
```

**Impact:**
- 💰 40-60% token reduction
- 📉 Lower API costs
- ⚡ Faster processing
- 🎯 Maintained quality (each agent gets optimal context)

**Rationale:**
- **Emotional (0)**: Needs fresh perspective, history adds bias
- **Critical/Optimistic (3)**: Focused analysis, minimal context needed
- **Factual (5)**: Recent context for data gathering
- **Creative (8)**: Broader context sparks connections
- **Synthesis (10)**: Needs maximum context to integrate all perspectives

### State Conversion Helpers

**Bridge Pattern:** Convert between session_state dict and typed state for Agno compatibility.

```python
def _get_typed_state_from_session(self, session_state: dict[str, Any]) -> MultiThinkingState:
    """Extract typed state from session_state dict."""
    return MultiThinkingState(
        current_strategy=session_state.get("current_strategy", "pending"),
        current_complexity_score=session_state.get("current_complexity_score", 0.0),
        thinking_sequence=session_state.get("thinking_sequence", []),
        # ... other fields
    )

def _save_typed_state_to_session(self, state: MultiThinkingState, session_state: dict[str, Any]) -> None:
    """Save typed state back to session_state dict."""
    session_state["current_strategy"] = state.current_strategy
    session_state["current_complexity_score"] = state.current_complexity_score
    # ... other fields
```

**Why:** Agno 2.2.12 still uses session_state internally; this provides type safety while maintaining compatibility.

**Security & Rate Limiting:**
- Prompt injection protection with regex patterns and Shannon entropy
- Request size validation (50KB max)
- Token bucket algorithm for burst protection (30 req/min, 500 req/hour)
- Concurrent request limiting (5 max)
- Comprehensive input sanitization with HTML escaping

## Common Issues

- **Circular imports** → Use `TYPE_CHECKING` or dynamic imports
- **Empty Agno content** → Check `StepOutput.success` and `session_state`
- **API key errors** → Ensure real tokens (GitHub needs 15+ unique chars)
- **ExaTools import errors** → Optional dependency, graceful degradation built-in