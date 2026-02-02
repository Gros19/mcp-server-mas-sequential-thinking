"""Unit tests for multi-thinking processor event metrics."""

from types import SimpleNamespace

from mcp_server_mas_sequential_thinking.processors.multi_thinking_processor import (
    MultiThinkingSequentialProcessor,
)
from mcp_server_mas_sequential_thinking.routing.workflow_state import (
    MultiThinkingState,
)


def test_record_token_usage_from_payload_updates_state():
    """Token usage should be recorded when metrics are available."""
    state = MultiThinkingState()
    payload = SimpleNamespace(
        metrics=SimpleNamespace(input_tokens=12, output_tokens=34)
    )

    MultiThinkingSequentialProcessor._record_token_usage_from_payload(
        state, "factual", payload
    )

    assert state.token_usage["factual"]["input_tokens"] == 12
    assert state.token_usage["factual"]["output_tokens"] == 34
    assert state.token_usage["factual"]["total_tokens"] == 46


def test_record_token_usage_from_payload_direct_fields():
    """Direct token fields should also be recognized."""
    state = MultiThinkingState()
    payload = SimpleNamespace(input_tokens=5, output_tokens=6)

    MultiThinkingSequentialProcessor._record_token_usage_from_payload(
        state, "critical", payload
    )

    assert state.token_usage["critical"]["total_tokens"] == 11
