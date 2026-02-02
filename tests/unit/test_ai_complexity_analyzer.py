"""Unit tests for AI complexity analyzer agent setup."""

from types import SimpleNamespace

from mcp_server_mas_sequential_thinking.routing import ai_complexity_analyzer


class DummyAgent:
    """Capture initialization values for assertions."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_get_agent_falls_back_to_enhanced_model(monkeypatch):
    """Analyzer should use enhanced model when create_agent_model is absent."""
    monkeypatch.setattr(ai_complexity_analyzer, "Agent", DummyAgent)
    monkeypatch.setattr(
        ai_complexity_analyzer,
        "create_learning_resources",
        lambda: SimpleNamespace(learning_machine="lm", db="db"),
    )

    model_config = SimpleNamespace(create_enhanced_model=lambda: "enhanced-model")
    analyzer = ai_complexity_analyzer.AIComplexityAnalyzer(model_config=model_config)

    agent = analyzer._get_agent()

    assert agent.kwargs["model"] == "enhanced-model"
    assert agent.kwargs["learning"] == "lm"
    assert agent.kwargs["db"] == "db"


def test_get_agent_prefers_create_agent_model_when_available(monkeypatch):
    """Analyzer should use dedicated agent model factory when present."""
    monkeypatch.setattr(ai_complexity_analyzer, "Agent", DummyAgent)
    monkeypatch.setattr(
        ai_complexity_analyzer,
        "create_learning_resources",
        lambda: SimpleNamespace(learning_machine="lm", db="db"),
    )

    model_config = SimpleNamespace(
        create_agent_model=lambda: "agent-model",
        create_enhanced_model=lambda: "enhanced-model",
    )
    analyzer = ai_complexity_analyzer.AIComplexityAnalyzer(model_config=model_config)

    agent = analyzer._get_agent()

    assert agent.kwargs["model"] == "agent-model"
