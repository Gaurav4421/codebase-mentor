"""
Deterministic extraction of "what symbols exist where" straight from
CodeChunk metadata -- no LLM call, so it's exact and free (same as the
notebook's `extract_capabilities`). This is the ground truth that
code_relationships.py's LLM-written summaries are grounded in, that
retrieval/hybrid_retriever.py labels chunks with, and that
code_intelligence.call_graph resolves call/inheritance edges against.

Covers functions, classes, methods, and (via chunker/parser's module-level
variable extraction) variables -- everything the ingestion layer turns into
a named CodeChunk.
"""
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ingestion.chunker import CodeChunk


@dataclass
class Symbol:
    name: str
    type: str  # e.g. "function_definition", "class_definition", "variable_definition"
    path: str
    start_line: int
    end_line: int
    parent_class: Optional[str] = None
    bases: List[str] = field(default_factory=list)


def extract_symbols(chunks: List[CodeChunk]) -> Dict[str, List[Symbol]]:
    """Group every named function/class/method/variable by the file it lives in."""
    by_file: Dict[str, List[Symbol]] = defaultdict(list)

    for chunk in chunks:
        if not chunk.name:
            continue
        by_file[chunk.path].append(Symbol(
            name=chunk.name,
            type=chunk.chunk_type,
            path=chunk.path,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            parent_class=chunk.parent_class,
            bases=chunk.bases,
        ))

    return dict(by_file)


def format_symbol_label(symbol: Symbol) -> str:
    """'Bar.baz' if baz is a method of Bar, else just 'baz'."""
    return f"{symbol.parent_class}.{symbol.name}" if symbol.parent_class else symbol.name


def symbol_node_id(symbol: Symbol) -> str:
    """Stable id used as a node key in the call graph and as a lookup key into
    the retrieval docs, e.g. 'pricing.py::Model.predict'."""
    return f"{symbol.path}::{format_symbol_label(symbol)}"


@dataclass
class SymbolIndex:
    """Queryable symbol table over a whole repo: name -> every occurrence
    (functions can be overloaded/redefined across files, so this is
    deliberately a list, not a single entry) plus the existing per-file view."""
    by_name: Dict[str, List[Symbol]]
    by_file: Dict[str, List[Symbol]]

    def find(self, name: str) -> List[Symbol]:
        return self.by_name.get(name, [])

    def find_qualified(self, qualified_name: str) -> List[Symbol]:
        """Look up 'Bar.baz' or a bare 'baz'."""
        if "." in qualified_name:
            parent, _, leaf = qualified_name.rpartition(".")
            return [s for s in self.by_name.get(leaf, []) if s.parent_class == parent]
        return self.find(qualified_name)

    def to_dict(self) -> dict:
        """First-occurrence view matching the requested
        {"name": {"type": ..., "file": ..., "line": ...}} shape. Use
        `.by_name` directly when you need every occurrence of a duplicated name."""
        return {
            name: {"type": occurrences[0].type, "file": occurrences[0].path, "line": occurrences[0].start_line}
            for name, occurrences in self.by_name.items()
        }


def build_symbol_index(chunks: List[CodeChunk]) -> SymbolIndex:
    by_file = extract_symbols(chunks)
    by_name: Dict[str, List[Symbol]] = defaultdict(list)
    for symbols in by_file.values():
        for symbol in symbols:
            by_name[symbol.name].append(symbol)
    return SymbolIndex(by_name=dict(by_name), by_file=by_file)
