"""
Ties the whole upgraded retrieval pipeline together:

    query
      |
    classify_query            (retrieval strategy: 6 intents)
      |
    generate_search_queries   (multi-query expansion, skipped for file_search)
      |
    +-- graph retrieval  (bug_debugging / dependency_question only) --+
    |                                                                  |
    +-- hybrid (BM25 + FAISS) per rewritten query, metadata-filtered --+
      |
    merge + dedup
      |
    rerank (CrossEncoder)
      |
    attach parent context     (architecture / documentation_generation only)

pipeline.py calls `adaptive_query(...)` where it used to call `hybrid_query`
directly; everything below it (prompt template, LLM call, sources footer) is
unchanged.
"""
import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx
from langchain_core.documents import Document

from code_intelligence.symbol_extractor import SymbolIndex
from llm.models import LLMClient
from retrieval.graph_retriever import retrieve_via_graph
from retrieval.hierarchy import CodeHierarchy
from retrieval.hybrid_retriever import HybridIndex, MetadataFilter, hybrid_query
from retrieval.query_classifier import QueryIntent, classify_query
from retrieval.query_rewriter import generate_search_queries
from retrieval.reranker import Reranker

logger = logging.getLogger(__name__)


_GRAPH_INTENTS = (QueryIntent.BUG_DEBUGGING, QueryIntent.DEPENDENCY_QUESTION)

_PARENT_CONTEXT_INTENTS = (QueryIntent.ARCHITECTURE, QueryIntent.DOCUMENTATION_GENERATION)

_KNOWN_EXTENSIONS = (
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs", ".cpp", ".c",
    ".h", ".hpp", ".html", ".css", ".json", ".yaml", ".yml", ".md",
)
_FILENAME_RE = re.compile(r"[\w\-/]+\.\w+")


@dataclass
class RetrievalContext:
    """Everything adaptive retrieval needs, built once at indexing time
    (see pipeline.py) and reused across every turn of a conversation."""
    hybrid_index: HybridIndex
    reranker: Reranker
    symbol_index: SymbolIndex
    call_graph: nx.MultiDiGraph
    hierarchy: CodeHierarchy
    docs_by_symbol_id: Dict[str, Document]
    llm: LLMClient


def infer_metadata_filter(query: str) -> Optional[MetadataFilter]:
    """Cheap heuristics for the common cases: the user names a file
    explicitly, or mentions a language by name. Anything more specific
    (module/symbol-type) is left for callers that already know it
    (e.g. file_search intent forcing symbol_type='file')."""
    filename_match = _FILENAME_RE.search(query)
    filename = None
    if filename_match and filename_match.group(0).lower().endswith(_KNOWN_EXTENSIONS):
        filename = filename_match.group(0)

    if filename:
        return MetadataFilter(filename=filename)
    return None


def _dedup(docs: List[Document]) -> List[Document]:
    seen = set()
    out = []
    for doc in docs:
        key = (doc.metadata["path"], doc.metadata["start_line"])
        if key not in seen:
            seen.add(key)
            out.append(doc)
    return out


def _attach_parent_context(doc: Document, hierarchy: CodeHierarchy) -> Document:
    meta = doc.metadata
    if not meta.get("name"):
        return doc
    label = f"{meta['parent_class']}.{meta['name']}" if meta.get("parent_class") else meta["name"]
    node_id = f"{meta['path']}::{label}"
    context = hierarchy.parent_context_text(node_id)
    if not context:
        return doc
    return Document(page_content=f"{context}\n\n{doc.page_content}", metadata=meta)


def adaptive_query(
    query: str,
    ctx: RetrievalContext,
    top_k: int = 5,
) -> Tuple[List[Document], QueryIntent, List[str]]:
    """Returns (documents, retrieval_intent, search_queries_used)."""
    intent = classify_query(query)
    search_queries = generate_search_queries(query, ctx.llm, intent)
    metadata_filter = infer_metadata_filter(query)

    graph_docs: List[Document] = []
    if intent in _GRAPH_INTENTS:
        graph_docs = retrieve_via_graph(
            query, ctx.symbol_index, ctx.call_graph, ctx.docs_by_symbol_id,
            include_callers=(intent == QueryIntent.DEPENDENCY_QUESTION),
        )
        if graph_docs:
            logger.info("Graph retrieval matched %d related symbols for: %s", len(graph_docs), query)

    hybrid_docs: List[Document] = []
    for search_query in search_queries:
        hybrid_docs.extend(
            hybrid_query(search_query, ctx.hybrid_index, ctx.reranker, top_k=top_k,
                         metadata_filter=metadata_filter, skip_rerank=True)
        )

    # Graph hits are structurally guaranteed relevant (they came from the
    # symbol the user actually named) -- put them first so they survive the
    # rerank's top_k truncation even when the hybrid side is noisy.
    merged = _dedup(graph_docs + hybrid_docs)

    final_top_k = top_k if intent != QueryIntent.ARCHITECTURE else max(top_k, 8)
    ranked = ctx.reranker.rerank(query, merged, top_k=final_top_k) if len(merged) > final_top_k else merged

    if intent in _PARENT_CONTEXT_INTENTS:
        ranked = [_attach_parent_context(d, ctx.hierarchy) for d in ranked]

    return ranked, intent, search_queries
