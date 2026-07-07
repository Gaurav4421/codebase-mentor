"""
Implements the "find predict() -> find what it calls -> retrieve that code"
path for bug-debugging and dependency questions. This is what lets the
mentor answer structurally instead of only by embedding similarity: a
question like "why does prediction fail?" should pull in `predict`'s own
body *and* the functions it calls (and, for dependency questions, its
callers too), not just whatever chunk happens to be semantically closest.

Token extraction from the query is a light regex pass (finding
identifier-shaped words) -- that's just "which words look like names", not
code parsing. All the actual code-structure understanding (who calls whom,
who inherits from whom) comes from the tree-sitter-built call graph, per the
"no regex for code understanding" requirement.
"""
import re
from typing import Dict, List

import networkx as nx
from langchain_core.documents import Document

from code_intelligence.call_graph import base_classes_of, callees_of, callers_of
from code_intelligence.symbol_extractor import SymbolIndex, symbol_node_id

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

_STOPWORDS = {
    "the", "why", "does", "what", "when", "where", "how", "fail", "fails",
    "failing", "error", "bug", "this", "that", "function", "method", "class",
    "call", "calls", "called", "code", "file", "and", "for", "with", "from",
}


def extract_candidate_names(query: str) -> List[str]:
    """Identifier-looking tokens from the query, longest/most-specific first,
    so ambiguous short words don't crowd out a real symbol name."""
    tokens = _TOKEN_RE.findall(query)
    candidates = [t for t in tokens if len(t) > 2 and t.lower() not in _STOPWORDS]
    return sorted(set(candidates), key=len, reverse=True)


def resolve_query_symbols(query: str, symbol_index: SymbolIndex, max_matches: int = 3):
    """Every symbol whose name appears as a token in the query, capped so a
    generic word doesn't pull in the whole repo."""
    matches = []
    for name in extract_candidate_names(query):
        matches.extend(symbol_index.find(name))
        if len(matches) >= max_matches:
            break
    return matches[:max_matches]


def graph_expand_node_ids(
    matched_symbols,
    call_graph: nx.MultiDiGraph,
    include_callers: bool = True,
    include_bases: bool = True,
) -> List[str]:
    """1-hop expansion from each matched symbol: itself, what it calls, what
    calls it (for dependency questions), and its base classes."""
    node_ids = set()
    for symbol in matched_symbols:
        node_id = symbol_node_id(symbol)
        if not call_graph.has_node(node_id):
            continue
        node_ids.add(node_id)
        node_ids.update(callees_of(call_graph, node_id))
        if include_callers:
            node_ids.update(callers_of(call_graph, node_id))
        if include_bases:
            node_ids.update(base_classes_of(call_graph, node_id))
    return list(node_ids)


def retrieve_via_graph(
    query: str,
    symbol_index: SymbolIndex,
    call_graph: nx.MultiDiGraph,
    docs_by_symbol_id: Dict[str, Document],
    include_callers: bool = True,
) -> List[Document]:
    """Full graph-retrieval path: query -> matched symbols -> graph
    neighborhood -> code documents. Returns [] (never raises) when nothing in
    the query matches a known symbol, so the caller can fall back to
    ordinary hybrid retrieval."""
    matches = resolve_query_symbols(query, symbol_index)
    if not matches:
        return []

    node_ids = graph_expand_node_ids(matches, call_graph, include_callers=include_callers)
    return [docs_by_symbol_id[n] for n in node_ids if n in docs_by_symbol_id]
