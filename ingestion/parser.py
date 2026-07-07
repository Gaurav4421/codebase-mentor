"""
Tree-sitter parsing: turns raw file content into a list of structural nodes
(functions/classes/methods/module-level variables + the "interesting"
statements inside them).

Extends the original notebook port with three additions needed for code
intelligence (symbol_index / call_graph), all done with tree-sitter node
structure rather than regex over source text:
  - call-node -> callee identifier extraction (InnerNode.callee)
  - class-node -> base-class identifier extraction (ParsedNode.bases)
  - module-level assignment/declaration -> "variable_definition" ParsedNode
"""
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from tree_sitter_language_pack import get_parser

logger = logging.getLogger(__name__)

TOP_LEVEL_TYPES = (
    "function_definition",
    "function_declaration",
    "method_definition",
    "class_definition",
    "class_declaration",
)

CLASS_TYPES = ("class_definition", "class_declaration")

BASE_CLASS_CONTAINER_TYPES = (
    "argument_list",      
    "superclasses",
    "superclass",        
    "class_heritage",     
    "base_class_clause", 
)

IMPORTANT_TYPES = (
    "if_statement",
    "for_statement",
    "while_statement",
    "try_statement",
    "return_statement",
    "call",
    "assignment",
)

MODULE_VARIABLE_TYPES = (
    "assignment",
    "variable_declaration",
    "lexical_declaration",
)

_CALLEE_IDENTIFIER_TYPES = ("identifier", "property_identifier", "field_identifier")


@dataclass
class InnerNode:
    type: str
    text: str
    start_line: int
    end_line: int
    callee: Optional[str] = None 


@dataclass
class ParsedNode:
    text: str
    type: str
    name: Optional[str]
    parent_class: Optional[str]
    start_line: int
    end_line: int
    child_nodes: List[InnerNode] = field(default_factory=list)
    bases: List[str] = field(default_factory=list)  


def _get_node_name(node, content: str) -> Optional[str]:
    for child in node.children:
        if child.type in ("identifier", "property_identifier", "type_identifier"):
            return content[child.start_byte:child.end_byte]
    return None


def _get_callee_name(node, content: str) -> Optional[str]:
    """For a `call` node, return the name actually being invoked: the last
    identifier-like token before the argument list (so `foo(...)` -> "foo",
    `self.bar(...)` -> "bar", `obj.ns.method(...)` -> "method")."""
    candidates = []

    def walk(n):
        if n.type == "argument_list" or n.type == "arguments":
            return 
        if n.type in _CALLEE_IDENTIFIER_TYPES:
            candidates.append(content[n.start_byte:n.end_byte])
        for child in n.children:
            walk(child)

    for child in node.children:
        if child.type in ("argument_list", "arguments"):
            break
        walk(child)

    return candidates[-1] if candidates else None


def _get_base_classes(node, content: str) -> List[str]:
    """For a class node, collect identifiers found inside its
    inheritance-clause child (see BASE_CLASS_CONTAINER_TYPES)."""
    bases = []
    for child in node.children:
        if child.type in BASE_CLASS_CONTAINER_TYPES:
            for sub in child.children:
                if sub.type in ("identifier", "type_identifier"):
                    bases.append(content[sub.start_byte:sub.end_byte])
    return bases


def _first_identifier(node, content: str, depth: int = 0, max_depth: int = 3) -> Optional[str]:
    """Shallow DFS for the first identifier-like token in `node`. Used to find
    an assignment's target name without a full per-language LHS resolver --
    good enough for `x = ...`, `x: int = ...`, `const x = ...`, `let x = ...`."""
    if depth > max_depth:
        return None
    for child in node.children:
        if child.type in ("identifier", "property_identifier"):
            return content[child.start_byte:child.end_byte]
    for child in node.children:
        found = _first_identifier(child, content, depth + 1, max_depth)
        if found:
            return found
    return None


def _collect_inner_nodes(node, content: str, out: List[InnerNode]) -> None:
    """Collect only meaningful logic nodes without duplicating chunks."""
    if node.type in IMPORTANT_TYPES:
        callee = _get_callee_name(node, content) if node.type == "call" else None
        out.append(InnerNode(
            type=node.type,
            text=content[node.start_byte:node.end_byte].strip(),
            start_line=node.start_point[0],
            end_line=node.end_point[0],
            callee=callee,
        ))
    for child in node.children:
        _collect_inner_nodes(child, content, out)


def parse_source(file_path: str, language: str, content: str) -> List[ParsedNode]:
    """Parse one file's content into ParsedNode structural chunks: functions,
    classes/methods (with bases + inner call nodes), and module-level
    variable definitions. Falls back to an empty list (never raises) on
    parser errors -- the caller (chunker.py) decides what to do when a file
    yields nothing (e.g. whole-file fallback)."""
    try:
        parser = get_parser(language)
        tree = parser.parse(bytes(content, "utf-8"))
    except Exception as e:
        logger.warning("tree-sitter failed to parse %s (%s): %s", file_path, language, e)
        return []

    results: List[ParsedNode] = []

    def extract(node, enclosing_class: Optional[str] = None, top_level: bool = True) -> None:
        if node.type in TOP_LEVEL_TYPES:
            snippet = content[node.start_byte:node.end_byte].strip()
            if snippet:
                inner_nodes: List[InnerNode] = []
                for child in node.children:
                    _collect_inner_nodes(child, content, inner_nodes)

                results.append(ParsedNode(
                    text=snippet,
                    type=node.type,
                    name=_get_node_name(node, content),
                    parent_class=enclosing_class,
                    start_line=node.start_point[0],
                    end_line=node.end_point[0],
                    child_nodes=inner_nodes,
                    bases=_get_base_classes(node, content) if node.type in CLASS_TYPES else [],
                ))
        elif node.type in MODULE_VARIABLE_TYPES and top_level:
            var_name = _first_identifier(node, content)
            if var_name:
                results.append(ParsedNode(
                    text=content[node.start_byte:node.end_byte].strip(),
                    type="variable_definition",
                    name=var_name,
                    parent_class=None,
                    start_line=node.start_point[0],
                    end_line=node.end_point[0],
                ))

        # If this node IS a class, its children should know their enclosing class.
        next_enclosing = _get_node_name(node, content) if node.type in CLASS_TYPES else enclosing_class
        # Once we're inside a function/class body, nothing below is "module level" anymore.
        next_top_level = top_level and node.type not in TOP_LEVEL_TYPES
        for child in node.children:
            extract(child, next_enclosing, next_top_level)

    extract(tree.root_node)
    return results
