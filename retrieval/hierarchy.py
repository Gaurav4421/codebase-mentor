"""
Repository -> File -> Class -> Function parent-child index.

Retrieval always happens at the function/method/variable-chunk level (that's
what's embedded and BM25-indexed), but some questions need the *enclosing*
context, not just the leaf: "what does this belong to", "how does this fit
into the module", documentation generation, etc. This module builds the
parent chain once at indexing time so a retrieved leaf chunk can be expanded
with its class/file/module/repo context on demand, instead of re-deriving it
per query.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from code_intelligence.symbol_extractor import Symbol, symbol_node_id


@dataclass
class HierarchyNode:
    id: str
    level: str  # "repository" | "module" | "file" | "class" | "function"
    label: str
    parent_id: Optional[str]
    summary: str = ""


@dataclass
class CodeHierarchy:
    repo_path: str
    nodes: Dict[str, HierarchyNode] = field(default_factory=dict)
    children: Dict[str, List[str]] = field(default_factory=dict)

    def parent_chain(self, node_id: str) -> List[HierarchyNode]:
        """Root-first chain of ancestors for `node_id` (not including itself)."""
        chain = []
        current = self.nodes.get(node_id)
        while current and current.parent_id:
            parent = self.nodes.get(current.parent_id)
            if parent is None:
                break
            chain.append(parent)
            current = parent
        return list(reversed(chain))

    def parent_context_text(self, node_id: str) -> str:
        """Short 'breadcrumb + summaries' string for prompt injection."""
        chain = self.parent_chain(node_id)
        if not chain:
            return ""
        lines = [f"{'  ' * i}- {n.level}: {n.label}" + (f" -- {n.summary}" if n.summary else "")
                 for i, n in enumerate(chain)]
        return "Enclosing context:\n" + "\n".join(lines)


def _module_of(path: str, repo_root: str) -> str:
    import os
    from pathlib import Path
    rel = os.path.relpath(path, repo_root)
    parts = Path(rel).parts
    return parts[0] if len(parts) > 1 else "(root)"


def build_hierarchy(
    repo_path: str,
    symbols_by_file: Dict[str, List[Symbol]],
    repo_summary: str,
    module_summaries: dict,
) -> CodeHierarchy:
    hierarchy = CodeHierarchy(repo_path=repo_path)

    repo_id = "repo"
    hierarchy.nodes[repo_id] = HierarchyNode(id=repo_id, level="repository", label=repo_path,
                                              parent_id=None, summary=repo_summary)

    module_ids: Dict[str, str] = {}
    for path in symbols_by_file:
        module_name = _module_of(path, repo_path)
        if module_name not in module_ids:
            module_id = f"module::{module_name}"
            module_ids[module_name] = module_id
            mod_summary = module_summaries.get(module_name)
            summary_text = getattr(mod_summary, "summary", "") if mod_summary else ""
            hierarchy.nodes[module_id] = HierarchyNode(
                id=module_id, level="module", label=module_name, parent_id=repo_id, summary=summary_text,
            )
            hierarchy.children.setdefault(repo_id, []).append(module_id)

    for path, symbols in symbols_by_file.items():
        module_id = module_ids[_module_of(path, repo_path)]
        file_id = f"file::{path}"
        hierarchy.nodes[file_id] = HierarchyNode(id=file_id, level="file", label=path, parent_id=module_id)
        hierarchy.children.setdefault(module_id, []).append(file_id)

        class_ids: Dict[str, str] = {}
        for symbol in symbols:
            if symbol.type in ("class_definition", "class_declaration"):
                class_node_id = symbol_node_id(symbol)
                class_ids[symbol.name] = class_node_id
                hierarchy.nodes[class_node_id] = HierarchyNode(
                    id=class_node_id, level="class", label=symbol.name, parent_id=file_id,
                )
                hierarchy.children.setdefault(file_id, []).append(class_node_id)

        for symbol in symbols:
            if symbol.type in ("class_definition", "class_declaration"):
                continue
            node_id = symbol_node_id(symbol)
            parent_id = class_ids.get(symbol.parent_class, file_id)
            level = "function" if not symbol.parent_class else "function"
            hierarchy.nodes[node_id] = HierarchyNode(id=node_id, level=level, label=symbol.name, parent_id=parent_id)
            hierarchy.children.setdefault(parent_id, []).append(node_id)

    return hierarchy
