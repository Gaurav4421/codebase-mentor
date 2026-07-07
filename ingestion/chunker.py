"""
Orchestrates ingestion.file_loader + ingestion.parser into the final unit of
retrieval: CodeChunk. This is the module ingestion/retrieval/code_intelligence
all agree on as the shared data model, replacing the notebook's untyped dicts.
"""
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from ingestion.file_loader import SourceFile, iter_source_files
from ingestion.parser import parse_source

logger = logging.getLogger(__name__)


@dataclass
class CodeChunk:
    path: str
    language: str
    chunk_type: str
    content: str
    start_line: int
    end_line: int
    name: Optional[str] = None
    parent_class: Optional[str] = None
    file_imports: List[str] = field(default_factory=list)
    child_nodes: List[dict] = field(default_factory=list)
    bases: List[str] = field(default_factory=list)


def _whole_file_chunk(source: SourceFile) -> CodeChunk:
    """Fallback when tree-sitter yields nothing for a file (parse error, or a
    language/file where nothing matched TOP_LEVEL_TYPES, e.g. a config file)."""
    return CodeChunk(
        path=source.path,
        language=source.language,
        chunk_type="file",
        content=source.content,
        start_line=0,
        end_line=len(source.content.splitlines()),
        file_imports=source.imports,
    )


def chunk_source_file(source: SourceFile) -> List[CodeChunk]:
    parsed_nodes = parse_source(source.path, source.language, source.content)

    if not parsed_nodes:
        return [_whole_file_chunk(source)]

    chunks = []
    for node in parsed_nodes:
        if not node.text:
            continue
        chunks.append(CodeChunk(
            path=source.path,
            language=source.language,
            chunk_type=node.type,
            content=node.text,
            start_line=node.start_line,
            end_line=node.end_line,
            name=node.name,
            parent_class=node.parent_class,
            file_imports=source.imports,
            child_nodes=[vars(n) for n in node.child_nodes],
            bases=node.bases,
        ))
    return chunks


def load_and_chunk_repo(repo_path: str) -> List[CodeChunk]:
    """Walk `repo_path` and return every CodeChunk across every supported file.
    One bad file degrades to a whole-file chunk (via parse_source's empty-list
    fallback) rather than aborting the whole repo's indexing."""
    all_chunks: List[CodeChunk] = []
    file_count = 0

    for source in iter_source_files(repo_path):
        file_count += 1
        all_chunks.extend(chunk_source_file(source))

    logger.info("Chunked %d files into %d chunks from %s", file_count, len(all_chunks), repo_path)

    if not all_chunks:
        raise ValueError(f"No supported source files found in repository: {repo_path}")

    return all_chunks
