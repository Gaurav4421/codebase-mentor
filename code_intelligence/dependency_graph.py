"""
Builds a best-effort, language-agnostic file-level dependency graph from the
import lines every CodeChunk already carries (`file_imports`). This is new
functionality relative to the notebook -- the notebook only ever displayed
imports as text inside a chunk's context block; nothing built a graph out of
them. It's deliberately conservative: raw import lines are always kept, and
resolution to an actual in-repo file is "best effort" (matched by filename
stem), not a full per-language import resolver -- that would need a real
import resolver per language, which is out of scope for a repo-agnostic tool.
"""
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set

from ingestion.chunker import CodeChunk


@dataclass
class DependencyGraph:
    # file path -> raw import lines found in that file
    raw_imports: Dict[str, List[str]] = field(default_factory=dict)
    # file path -> set of other in-repo file paths it appears to depend on
    edges: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))

    def dependents_of(self, file_path: str) -> List[str]:
        """Files that import `file_path` (reverse edge lookup)."""
        return [src for src, targets in self.edges.items() if file_path in targets]


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def build_dependency_graph(chunks: List[CodeChunk]) -> DependencyGraph:
    graph = DependencyGraph()

    # One row of raw imports per file (dedup: multiple chunks per file all carry
    # the same file_imports list).
    for chunk in chunks:
        graph.raw_imports.setdefault(chunk.path, chunk.file_imports)

    all_paths = list(graph.raw_imports.keys())
    stem_to_paths: Dict[str, List[str]] = defaultdict(list)
    for path in all_paths:
        stem_to_paths[_stem(path)].append(path)

    for path, imports in graph.raw_imports.items():
        for line in imports:
            # Pull out the module/file token being imported, tolerant of the
            # handful of import styles extract_file_imports() recognizes.
            tokens = (
                line.replace(",", " ")
                .replace("{", " ").replace("}", " ")
                .replace("(", " ").replace(")", " ")
                .replace(";", " ")
                .split()
            )
            candidates = [t.strip("'\"./") for t in tokens if t not in
                          ("import", "from", "require", "using", "#include", "as")]

            for candidate in candidates:
                # match against the last path segment of dotted/slashed module names
                key = candidate.split(".")[-1].split("/")[-1]
                for target_path in stem_to_paths.get(key, []):
                    if target_path != path:
                        graph.edges[path].add(target_path)

    return graph
