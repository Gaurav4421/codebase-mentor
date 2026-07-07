import sys
from pathlib import Path

import pytest

# Allow `import pipeline`, `import ingestion.x`, etc. when pytest is run from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def tiny_repo(tmp_path: Path) -> Path:
    """A minimal, deterministic 2-file Python 'repository' on disk, used by
    ingestion/chunker/pipeline tests instead of relying on any real project
    being present in the test environment."""
    (tmp_path / "utils.py").write_text(
        '''"""Small math helpers."""


def add(a, b):
    """Return a + b."""
    return a + b


class Calculator:
    """A tiny calculator."""

    def __init__(self, start=0):
        self.value = start

    def add(self, amount):
        self.value = add(self.value, amount)
        return self.value
'''
    )
    (tmp_path / "main.py").write_text(
        '''"""Entry point that uses utils.Calculator."""
from utils import Calculator


def run():
    calc = Calculator()
    calc.add(5)
    return calc.value
'''
    )
    return tmp_path
