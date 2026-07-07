"""
CodebaseMentor: the single place that wires ingestion -> code_intelligence ->
retrieval -> llm -> memory together. app/streamlit_app.py and
evaluation/benchmark.py both depend on this instead of duplicating the wiring.

Upgraded from text-based RAG to a code-intelligence-aware pipeline: indexing
now also builds a SymbolIndex, a NetworkX call graph (calls/inherits/imports),
and a Repository->File->Class->Function hierarchy; querying now goes through
retrieval.adaptive_retriever instead of calling hybrid_query directly.
"""
import logging
from dataclasses import dataclass
from typing import List, Tuple

from langchain_core.documents import Document

from code_intelligence.call_graph import build_call_graph
from code_intelligence.code_relationships import RepositoryIntelligence, generate_repository_intelligence
from code_intelligence.dependency_graph import DependencyGraph, build_dependency_graph
from code_intelligence.symbol_extractor import SymbolIndex, build_symbol_index
from ingestion.chunker import CodeChunk, load_and_chunk_repo
from llm.models import LLMClient
from llm.prompts import PROMPT_TEMPLATES, build_context, build_sources_footer, classify_intent
from memory.conversation import ConversationMemory
from retrieval.adaptive_retriever import RetrievalContext, adaptive_query
from retrieval.hierarchy import CodeHierarchy, build_hierarchy
from retrieval.hybrid_retriever import build_docs_by_symbol_id, build_indexes
from retrieval.reranker import Reranker

logger = logging.getLogger(__name__)


@dataclass
class RepoIndex:
    """Everything produced once per repo, at indexing time."""
    repo_path: str
    chunks: List[CodeChunk]
    intelligence: RepositoryIntelligence
    dependency_graph: DependencyGraph
    symbol_index: SymbolIndex
    hierarchy: CodeHierarchy
    retrieval_ctx: RetrievalContext


class CodebaseMentor:
    def __init__(self, llm: LLMClient, reranker: Reranker = None):
        self.llm = llm
        self.reranker = reranker or Reranker()
        self.index: RepoIndex = None

    def index_repository(self, repo_path: str) -> RepoIndex:
        logger.info("Indexing repository: %s", repo_path)
        chunks = load_and_chunk_repo(repo_path)

        hybrid_index = build_indexes(chunks)
        symbol_index = build_symbol_index(chunks)
        intelligence = generate_repository_intelligence(repo_path, symbol_index.by_file, self.llm)
        dep_graph = build_dependency_graph(chunks)
        call_graph = build_call_graph(chunks, symbol_index.by_file, dep_graph)
        hierarchy = build_hierarchy(repo_path, symbol_index.by_file, intelligence.repo_summary, intelligence.modules)
        docs_by_symbol_id = build_docs_by_symbol_id(hybrid_index.docs)

        retrieval_ctx = RetrievalContext(
            hybrid_index=hybrid_index,
            reranker=self.reranker,
            symbol_index=symbol_index,
            call_graph=call_graph,
            hierarchy=hierarchy,
            docs_by_symbol_id=docs_by_symbol_id,
            llm=self.llm,
        )

        self.index = RepoIndex(
            repo_path=repo_path,
            chunks=chunks,
            intelligence=intelligence,
            dependency_graph=dep_graph,
            symbol_index=symbol_index,
            hierarchy=hierarchy,
            retrieval_ctx=retrieval_ctx,
        )
        return self.index

    def new_conversation(self, max_turns: int = 6) -> ConversationMemory:
        return ConversationMemory(self.llm, max_turns=max_turns)

    def ask(self, query: str, memory: ConversationMemory) -> Tuple[str, str, List[Document]]:
        """Answers one turn. Returns (final_answer_with_sources, prompt_intent, retrieved_docs).
        Does NOT mutate `memory` -- the caller adds turns + trims afterward
        (mirrors the notebook's loop and keeps this method side-effect-free/testable).
        """
        if self.index is None:
            raise RuntimeError("index_repository() must be called before ask()")

        retrieval_query = memory.resolve_followup_query(query)

        results, retrieval_intent, search_queries = adaptive_query(retrieval_query, self.index.retrieval_ctx)
        logger.info("retrieval_intent=%s search_queries=%s", retrieval_intent, search_queries)

        # Prompt-template selection stays keyword-based on the same query, via
        # llm.prompts's own (separate) classifier -- see retrieval/query_classifier.py
        # docstring for why these two classifications are intentionally distinct.
        prompt_intent = classify_intent(retrieval_query)
        template = PROMPT_TEMPLATES[prompt_intent]

        context = build_context(
            results, self.index.intelligence.repo_summary,
            self.index.intelligence.modules, self.index.repo_path,
        )

        messages = template.format_messages(
            chat_history=memory.as_prompt_history(), context=context, question=query,
        )
        prompt = "\n\n".join(f"[{m.type.upper()}]\n{m.content}" for m in messages)

        answer = self.llm.generate(prompt)
        final_answer = f"{answer}\n\n{build_sources_footer(results)}"

        return final_answer, prompt_intent, results
