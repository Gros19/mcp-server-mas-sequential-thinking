"""Learning resource helpers for Agno learning features."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from agno.db.sqlite import SqliteDb
from agno.learn import LearningMachine


@dataclass(frozen=True)
class LearningResources:
    """Container for shared learning resources."""

    learning_machine: LearningMachine
    db: SqliteDb


@lru_cache(maxsize=1)
def _create_default_learning_resources() -> LearningResources:
    return _create_learning_resources("~/.sequential_thinking/learning.db")


def _create_learning_resources(db_file: str) -> LearningResources:
    db_path = Path(db_file).expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    db = SqliteDb(db_file=str(db_path))
    learning_machine = LearningMachine(db=db)
    return LearningResources(learning_machine=learning_machine, db=db)


def create_learning_resources(db_file: str | None = None) -> LearningResources:
    """Create learning resources with a default local SQLite backend."""
    if db_file is None:
        return _create_default_learning_resources()
    return _create_learning_resources(db_file)
