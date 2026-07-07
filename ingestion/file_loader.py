"""
Repo-walking and raw file I/O. Deliberately has zero knowledge of tree-sitter or
chunking -- this module's only job is "here are the readable, supported source
files in this repo, and here's each one's raw content + import lines."
"""
import logging
import os
from dataclasses import dataclass, field
from typing import Iterator, List

from ingestion.language_detector import detect_language

logger = logging.getLogger(__name__)

DEFAULT_IGNORE_DIRS = {
    ".git", "node_modules", "venv", ".venv", "__pycache__", "dist", "build",
    ".mypy_cache", ".pytest_cache", "target", ".idea", ".vscode", "egg-info",
}

IMPORT_PREFIXES = ("import ", "from ", "require(", "using ", "#include")


@dataclass
class SourceFile:
    path: str
    language: str
    content: str
    imports: List[str] = field(default_factory=list)


def extract_file_imports(content: str, max_lines: int = 40) -> List[str]:
    """Cheap, language-agnostic import-line scan over the top of a file. Not a
    real parser -- just enough signal for dependency_graph and repo summaries."""
    imports = []
    for line in content.splitlines()[:max_lines]:
        stripped = line.strip()
        if stripped.startswith(IMPORT_PREFIXES):
            imports.append(stripped)
    return imports


def iter_source_files(
    repo_path: str,
    ignore_dirs: set = DEFAULT_IGNORE_DIRS,
) -> Iterator[SourceFile]:
    """Walk `repo_path`, yielding a SourceFile for every readable, supported file.
    Unreadable / binary / empty files are skipped and logged at debug level
    rather than raising -- one bad file should never abort indexing a whole repo.
    """
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in ignore_dirs]

        for filename in files:
            full_path = os.path.join(root, filename)
            language = detect_language(full_path)
            if not language:
                continue

            try:
                with open(full_path, "r", encoding="utf-8", errors="ignore") as fh:
                    content = fh.read()
            except OSError as e:
                logger.debug("Skipping unreadable file %s: %s", full_path, e)
                continue

            if not content.strip():
                continue

            yield SourceFile(
                path=full_path,
                language=language,
                content=content,
                imports=extract_file_imports(content),
            )
