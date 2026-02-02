"""Unit tests for learning resource initialization."""

from mcp_server_mas_sequential_thinking.infrastructure.learning_resources import (
    create_learning_resources,
)


def test_create_learning_resources_uses_valid_learning_machine_signature(tmp_path):
    """Learning resources should initialize without unsupported constructor args."""
    db_file = tmp_path / "learning.db"
    resources = create_learning_resources(str(db_file))

    assert db_file.exists() or db_file.parent.exists()
    assert resources.learning_machine is not None
    assert resources.db is not None


def test_create_learning_resources_reuses_default_instance():
    """Default resources should be cached to keep shared learning state."""
    first = create_learning_resources()
    second = create_learning_resources()

    assert first is second
