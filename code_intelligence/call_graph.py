"""
Builds a single NetworkX graph tying together everything code_intelligence
knows about a repo:

  file --contains--> symbol (function/class/method/variable)
  file --imports-->  file            (from dependency_graph's import scan)
  symbol --calls-->  symbol          (from parser.py's tree-sitter call nodes)
  class  --inherits--> base class    (from parser.py's tree-sitter class bases)

This is what lets a query like "why does prediction fail?" turn into "find
predict(), find what it calls, retrieve that code" instead of relying on
embedding similarity alone.

Call/inheritance resolution is name-based best-effort (same as
dependency_graph's import resolution): tree-sitter gives us the exact callee
identifier, but resolving `self.bar()` to a specific class's `bar` method
without full type inference is out of scope for a repo-agnostic tool, so we
prefer a same-file / same-class match and fall back to "any symbol with this
name" when that's ambiguous.
"""
import logging
from collections import defaultdict
from typing import Dict, List, Optional

import networkx as nx

from code_intelligence.dependency_graph import DependencyGraph
from code_intelligence.symbol_extractor import Symbol, format_symbol_label, symbol_node_id
from ingestion.chunker import CodeChunk

logger = logging.getLogger(__name__)


def _resolve_callee(
    callee_name: str,
    caller: Symbol,
    name_index: Dict[str, List[Symbol]],
) -> Optional[Symbol]:
    """Best-effort resolution of a call-site identifier to a symbol:
    1. a method of the same class (self.foo() / this.foo())
    2. any symbol with that name defined in the same file
    3. any symbol with that name anywhere in the repo (first match)
    """
    candidates = name_index.get(callee_name, [])
    if not candidates:
        return None

    if caller.parent_class:
        same_class = [c for c in candidates if c.parent_class == caller.parent_class and c.path == caller.path]
        if same_class:
            return same_class[0]

    same_file = [c for c in candidates if c.path == caller.path]
    if same_file:
        return same_file[0]

    return candidates[0]


def _resolve_base_class(base_name: str, cls: Symbol, name_index: Dict[str, List[Symbol]]) -> Optional[Symbol]:
    candidates = [c for c in name_index.get(base_name, []) if c.type in ("class_definition", "class_declaration")]
    if not candidates:
        return None
    same_file = [c for c in candidates if c.path == cls.path]
    return same_file[0] if same_file else candidates[0]


def build_call_graph(
    chunks: List[CodeChunk],
    symbols_by_file: Dict[str, List[Symbol]],
    dependency_graph: DependencyGraph,
) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()

    name_index: Dict[str, List[Symbol]] = defaultdict(list)
    for symbols in symbols_by_file.values():
        for symbol in symbols:
            name_index[symbol.name].append(symbol)

    # Nodes: files + symbols, and file --contains--> symbol edges.
    for path, symbols in symbols_by_file.items():
        graph.add_node(path, kind="file")
        for symbol in symbols:
            node_id = symbol_node_id(symbol)
            graph.add_node(
                node_id, kind=symbol.type, name=symbol.name, file=symbol.path,
                line=symbol.start_line, label=format_symbol_label(symbol),
            )
            graph.add_edge(path, node_id, relation="contains")

    # file --imports--> file
    for src, targets in dependency_graph.edges.items():
        for tgt in targets:
            if graph.has_node(src) and graph.has_node(tgt):
                graph.add_edge(src, tgt, relation="imports")

    # symbol --calls--> symbol, class --inherits--> class (from each chunk's
    # tree-sitter-collected call sites / base classes).
    chunks_by_symbol: Dict[str, CodeChunk] = {}
    for chunk in chunks:
        if chunk.name:
            chunks_by_symbol[f"{chunk.path}::{(chunk.parent_class + '.' if chunk.parent_class else '') + chunk.name}"] = chunk

    for path, symbols in symbols_by_file.items():
        for symbol in symbols:
            node_id = symbol_node_id(symbol)
            chunk = chunks_by_symbol.get(node_id)
            if chunk is None:
                continue

            for inner in chunk.child_nodes:
                if inner.get("type") != "call" or not inner.get("callee"):
                    continue
                target = _resolve_callee(inner["callee"], symbol, name_index)
                if target is not None:
                    graph.add_edge(node_id, symbol_node_id(target), relation="calls", callee_name=inner["callee"])

            for base_name in chunk.bases:
                base_symbol = _resolve_base_class(base_name, symbol, name_index)
                if base_symbol is not None:
                    graph.add_edge(node_id, symbol_node_id(base_symbol), relation="inherits")

    logger.info(
        "Built call graph: %d nodes, %d edges", graph.number_of_nodes(), graph.number_of_edges(),
    )
    return graph


def callees_of(graph: nx.MultiDiGraph, node_id: str) -> List[str]:
    if not graph.has_node(node_id):
        return []
    return [t for _, t, d in graph.out_edges(node_id, data=True) if d.get("relation") == "calls"]


def callers_of(graph: nx.MultiDiGraph, node_id: str) -> List[str]:
    if not graph.has_node(node_id):
        return []
    return [s for s, _, d in graph.in_edges(node_id, data=True) if d.get("relation") == "calls"]


def dependents_of_file(graph: nx.MultiDiGraph, path: str) -> List[str]:
    """Files that import `path`."""
    if not graph.has_node(path):
        return []
    return [s for s, _, d in graph.in_edges(path, data=True) if d.get("relation") == "imports"]


def base_classes_of(graph: nx.MultiDiGraph, node_id: str) -> List[str]:
    if not graph.has_node(node_id):
        return []
    return [t for _, t, d in graph.out_edges(node_id, data=True) if d.get("relation") == "inherits"]


def neighborhood(graph: nx.MultiDiGraph, node_id: str, hops: int = 1) -> List[str]:
    """All symbol/file node ids reachable within `hops` steps, in either
    direction, along calls/inherits/imports/contains edges. Used to pull the
    surrounding context for a matched symbol (its callees, its callers, and
    the file that contains it)."""
    if not graph.has_node(node_id):
        return []
    undirected = graph.to_undirected(as_view=True)
    lengths = nx.single_source_shortest_path_length(undirected, node_id, cutoff=hops)
    return list(lengths.keys())
