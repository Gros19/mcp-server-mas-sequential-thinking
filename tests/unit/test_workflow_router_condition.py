"""Unit tests for workflow routing structure."""

from mcp_server_mas_sequential_thinking.routing import agno_workflow_router


class DummyProcessor:
    """Lightweight processor stub to avoid heavy initialization in tests."""

    def __init__(self):
        self.processed = True


def test_workflow_uses_fixed_full_sequence(monkeypatch):
    """Workflow should always run complexity analysis + full sequence."""
    monkeypatch.setattr(
        agno_workflow_router,
        "MultiThinkingSequentialProcessor",
        DummyProcessor,
    )

    workflow_router = agno_workflow_router.MultiThinkingWorkflowRouter()

    steps = workflow_router.workflow.steps
    assert len(steps) == 2
    assert steps[0].name == "complexity_analysis"
    assert steps[1].name == "full_sequence"
