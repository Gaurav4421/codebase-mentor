"""
Composes retrieval.vector_store + retrieval.bm25 into an EnsembleRetriever at
index time, and reproduces the notebook's `query_repo` adaptive-ordering /
dedup / rerank pipeline at query time.

This is the one module allowed to know about both vector_store.py and bm25.py
-- everything else (pipeline.py, app, benchmark, adaptive_retriever) only
talks to this module's `HybridIndex` and `hybrid_query`.
"""
import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document

from ingestion.chunker import CodeChunk
from retrieval.bm25 import build_bm25_retriever
from retrieval.reranker import Reranker
from retrieval.vector_store import build_vector_index, chunk_to_document, _embed_text_for_chunk

logger = logging.getLogger(__name__)


@dataclass
class HybridIndex:
    vector_retriever: object
    bm_retriever: object
    ensemble_retriever: EnsembleRetriever
    docs: List[Document] 


def _normalize_for_dedup(text: str) -> str:
    return " ".join(text.split())


def build_indexes(chunks: List[CodeChunk], vector_k: int = 12, bm25_k: int = 12) -> HybridIndex:
    """Builds the raw-code documents (deduped by content hash) plus the
    enriched embed-documents, then constructs FAISS + BM25 + an
    EnsembleRetriever over them -- identical weighting/behavior to the
    notebook (0.5/0.5, MMR search on the vector side)."""
    seen_hashes = set()
    docs: List[Document] = []
    embed_docs: List[Document] = []

    for chunk in chunks:
        norm = _normalize_for_dedup(chunk.content)
        content_hash = hashlib.sha1(norm.encode("utf-8")).hexdigest()
        dedup_key = (chunk.path, content_hash)
        if dedup_key in seen_hashes:
            continue
        seen_hashes.add(dedup_key)

        doc = chunk_to_document(chunk)
        docs.append(doc)
        embed_docs.append(Document(page_content=_embed_text_for_chunk(chunk), metadata=doc.metadata))

    vector_index = build_vector_index(embed_docs)
    vector_retriever = vector_index.as_retriever(k=vector_k, search_type="mmr")
    bm_retriever = build_bm25_retriever(docs, k=bm25_k)

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm_retriever, vector_retriever],
        weights=[0.5, 0.5],
    )

    logger.info("Built hybrid index: %d raw docs, %d embed docs", len(docs), len(embed_docs))
    return HybridIndex(
        vector_retriever=vector_retriever,
        bm_retriever=bm_retriever,
        ensemble_retriever=ensemble_retriever,
        docs=docs,
    )


def build_docs_by_symbol_id(docs: List[Document]) -> Dict[str, Document]:
    """Index HybridIndex.docs by the same 'path::Class.method' id used in the
    call graph and symbol index, so graph_retriever can turn a matched symbol
    directly into its code document."""
    by_id: Dict[str, Document] = {}
    for doc in docs:
        meta = doc.metadata
        if not meta.get("name"):
            continue
        label = f"{meta['parent_class']}.{meta['name']}" if meta.get("parent_class") else meta["name"]
        by_id[f"{meta['path']}::{label}"] = doc
    return by_id


@dataclass
class MetadataFilter:
    filename: Optional[str] = None
    language: Optional[str] = None
    module: Optional[str] = None
    symbol_type: Optional[str] = None


def matches_filter(doc: Document, filt: MetadataFilter) -> bool:
    meta = doc.metadata
    if filt.filename and filt.filename.lower() not in meta.get("path", "").lower():
        return False
    if filt.language and meta.get("language") != filt.language:
        return False
    if filt.module and filt.module.lower() not in meta.get("path", "").lower():
        return False
    if filt.symbol_type and meta.get("type") != filt.symbol_type:
        return False
    return True


def _restore_raw_content(doc: Document) -> Document:
    """Vector-store hits carry the enriched embed_text as page_content (that's
    what got embedded) -- swap it back for the raw code before this reaches
    the reranker or the LLM."""
    raw = doc.metadata.get("code")
    if raw is not None and doc.page_content != raw:
        return Document(page_content=raw, metadata=doc.metadata)
    return doc


def _looks_like_keyword_query(query: str) -> bool:
    """Heuristic: identifiers / quoted strings / short queries -> the user is
    probably searching for something exact, so lean on BM25 first."""
    return bool(re.search(r'[_.]|["\']|\bdef\b|\bclass\b', query)) or query.strip().count(" ") <= 2


def hybrid_query(
    query: str,
    index: HybridIndex,
    reranker: Reranker,
    top_k: int = 5,
    metadata_filter: Optional[MetadataFilter] = None,
    skip_rerank: bool = False,
) -> List[Document]:
    """
    - Restore raw code (not the embed-context header) before dedup/rerank.
    - Adaptive ordering: keyword-ish queries favor BM25 hits first; conceptual
      questions favor vector hits first.
    - Optional metadata filter (filename/language/module/symbol type) applied
      before reranking, so filtered-out candidates don't crowd out real hits.
    - Reranker gets a slightly larger top_k when there are more candidates,
      instead of always truncating to a fixed 5.
    """
    vec_docs = [_restore_raw_content(d) for d in index.vector_retriever.invoke(query)]
    bm_docs = index.bm_retriever.invoke(query)

    ordered = (bm_docs + vec_docs) if _looks_like_keyword_query(query) else (vec_docs + bm_docs)

    merged_docs = []
    seen = set()
    for doc in ordered:
        key = (doc.metadata["path"], doc.metadata["start_line"])
        if key not in seen:
            seen.add(key)
            merged_docs.append(doc)

    if metadata_filter is not None:
        merged_docs = [d for d in merged_docs if matches_filter(d, metadata_filter)]

    merged_docs = merged_docs[:20]

    if skip_rerank:
        return merged_docs[:top_k]

    effective_top_k = top_k if len(merged_docs) <= 8 else min(top_k + 2, 8)
    return reranker.rerank(query, merged_docs, top_k=effective_top_k)
