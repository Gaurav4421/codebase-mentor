"""
BM25 keyword retrieval over the raw (non-enriched) code documents. Kept as its
own module so hybrid_retriever.py can compose it with vector_store.py without
either one knowing the other exists.
"""
import logging
from typing import List

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

DEFAULT_K = 12


def build_bm25_retriever(docs: List[Document], k: int = DEFAULT_K) -> BM25Retriever:
    retriever = BM25Retriever.from_documents(docs)
    retriever.k = k
    logger.info("Built BM25 index over %d documents", len(docs))
    return retriever
