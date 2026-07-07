"""
Maps file extensions to the tree-sitter grammar name used for parsing/chunking.

Kept intentionally dumb and data-only (extension -> language string) so adding
support for a new language is a one-line change here, not a change to parsing
logic anywhere else.
"""
from pathlib import Path
from typing import Optional

LANGUAGE_MAP = {
    # Python
    ".py": "python",
    ".pyi": "python",
    # JavaScript / TypeScript
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    # Web
    ".html": "html",
    ".css": "css",
    ".scss": "css",
    # Systems
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".c": "c",
    ".h": "cpp",
    ".hpp": "cpp",
    # JVM
    ".java": "java",
    # Go / Rust
    ".go": "go",
    ".rs": "rust",
    # Config / docs (still useful context for RAG even though they're not "code")
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".md": "markdown",
}


def detect_language(file_path: str) -> Optional[str]:
    """Return the tree-sitter language name for a file, or None if unsupported."""
    ext = Path(file_path).suffix.lower()
    return LANGUAGE_MAP.get(ext)


def is_supported(file_path: str) -> bool:
    return detect_language(file_path) is not None
