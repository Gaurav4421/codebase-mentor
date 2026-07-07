from code_intelligence.symbol_extractor import (
    build_symbol_index,
    format_symbol_label,
    symbol_node_id,
)
from ingestion.chunker import CodeChunk


def make_chunks():
    return [
        CodeChunk(
            path="calc.py", language="python", chunk_type="function_definition",
            content="def add(a, b): return a + b", start_line=1, end_line=1, name="add",
        ),
        CodeChunk(
            path="calc.py", language="python", chunk_type="class_definition",
            content="class Calculator: ...", start_line=3, end_line=10, name="Calculator",
            bases=["BaseCalculator"],
        ),
        CodeChunk(
            path="calc.py", language="python", chunk_type="method_definition",
            content="def add(self, x): ...", start_line=5, end_line=6, name="add",
            parent_class="Calculator",
        ),
        # Unnamed chunk (e.g. a whole-file fallback) must be skipped.
        CodeChunk(
            path="config.json", language="json", chunk_type="file",
            content="{}", start_line=0, end_line=1,
        ),
    ]


def test_build_symbol_index_groups_by_file_and_name():
    index = build_symbol_index(make_chunks())

    assert set(index.by_file.keys()) == {"calc.py"}
    assert len(index.by_file["calc.py"]) == 3

    # "add" appears twice: the free function and the method -- both preserved.
    assert len(index.by_name["add"]) == 2


def test_find_qualified_disambiguates_method_from_function():
    index = build_symbol_index(make_chunks())

    method_matches = index.find_qualified("Calculator.add")
    assert len(method_matches) == 1
    assert method_matches[0].parent_class == "Calculator"

    bare_matches = index.find_qualified("add")
    assert len(bare_matches) == 2


def test_format_symbol_label_and_node_id():
    index = build_symbol_index(make_chunks())
    method = index.find_qualified("Calculator.add")[0]
    free_fn = [s for s in index.by_name["add"] if s.parent_class is None][0]

    assert format_symbol_label(method) == "Calculator.add"
    assert format_symbol_label(free_fn) == "add"
    assert symbol_node_id(method) == "calc.py::Calculator.add"


def test_unnamed_chunks_are_excluded():
    index = build_symbol_index(make_chunks())
    assert "config.json" not in index.by_file
