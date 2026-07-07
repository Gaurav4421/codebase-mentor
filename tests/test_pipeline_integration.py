"""
Full-stack integration test: real chunking (tree-sitter), real FAISS +
BM25 indexing, real reranker/embeddings -- only the LLM is faked (via
FakeLLMClient), so the test is deterministic and needs no API key.

Excluded from the default `pytest` run (see pyproject.toml) because it
downloads real HuggingFace models on first run. Run explicitly with:

    pytest -m integration
"""
import pytest

from llm.models import FakeLLMClient
from pipeline import CodebaseMentor


@pytest.mark.integration
def test_index_and_ask_end_to_end(tiny_repo):
    mentor = CodebaseMentor(llm=FakeLLMClient(response="This is a mocked answer about Calculator.add."))

    repo_index = mentor.index_repository(str(tiny_repo))
    assert len(repo_index.chunks) > 0
    assert "Calculator" in repo_index.symbol_index.by_name or any(
        s.name == "Calculator" for symbols in repo_index.symbol_index.by_file.values() for s in symbols
    )

    memory = mentor.new_conversation()
    answer, intent, results = mentor.ask("What does the Calculator class do?", memory)

    assert "mocked answer" in answer
    assert intent  # a prompt intent was classified
    assert isinstance(results, list)
