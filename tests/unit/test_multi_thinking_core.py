"""Unit tests for multi-thinking core agent configuration."""

from types import SimpleNamespace

from mcp_server_mas_sequential_thinking.processors import multi_thinking_core


class DummyAgent:
    """Capture Agent initialization arguments for assertions."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.instructions = kwargs.get("instructions", [])
        self.enable_user_memories = kwargs.get("enable_user_memories", False)
        self.update_memory_on_run = kwargs.get("update_memory_on_run", False)


def test_synthesis_agent_includes_reasoning_tools(monkeypatch):
    """Synthesis agents should enable reasoning instructions by default."""
    monkeypatch.setattr(multi_thinking_core, "Agent", DummyAgent)

    factory = multi_thinking_core.MultiThinkingAgentFactory(
        learning_machine="learning",
        learning_db="db",
    )
    agent = factory.create_thinking_agent(
        multi_thinking_core.ThinkingDirection.SYNTHESIS,
        model=SimpleNamespace(),
    )

    tools = agent.kwargs.get("tools") or []
    assert any(getattr(tool, "add_instructions", False) for tool in tools)


def test_factory_passes_learning_and_culture_flags(monkeypatch):
    """Agents should receive learning resources while culture updates stay opt-in."""
    monkeypatch.setattr(multi_thinking_core, "Agent", DummyAgent)

    factory = multi_thinking_core.MultiThinkingAgentFactory(
        learning_machine="learning",
        learning_db="db",
    )
    agent = factory.create_thinking_agent(
        multi_thinking_core.ThinkingDirection.FACTUAL,
        model=SimpleNamespace(),
    )

    assert agent.kwargs["learning"] == "learning"
    assert agent.kwargs["db"] == "db"
    assert agent.kwargs["add_culture_to_context"] is False
    assert agent.kwargs["update_cultural_knowledge"] is False


def test_factory_allows_explicit_culture_learning_enable(monkeypatch):
    """Callers can still enable culture learning explicitly."""
    monkeypatch.setattr(multi_thinking_core, "Agent", DummyAgent)

    factory = multi_thinking_core.MultiThinkingAgentFactory(
        learning_machine="learning",
        learning_db="db",
    )
    agent = factory.create_thinking_agent(
        multi_thinking_core.ThinkingDirection.FACTUAL,
        model=SimpleNamespace(),
        add_culture_to_context=True,
        update_cultural_knowledge=True,
    )

    assert agent.kwargs["add_culture_to_context"] is True
    assert agent.kwargs["update_cultural_knowledge"] is True


def test_factory_reads_culture_learning_env_flag(monkeypatch):
    """Environment flag should enable culture learning defaults."""
    monkeypatch.setattr(multi_thinking_core, "Agent", DummyAgent)
    monkeypatch.setenv("SEQUENTIAL_THINKING_ENABLE_CULTURE_LEARNING", "true")

    factory = multi_thinking_core.MultiThinkingAgentFactory(
        learning_machine="learning",
        learning_db="db",
    )
    agent = factory.create_thinking_agent(
        multi_thinking_core.ThinkingDirection.FACTUAL,
        model=SimpleNamespace(),
    )

    assert agent.kwargs["add_culture_to_context"] is True
    assert agent.kwargs["update_cultural_knowledge"] is True
