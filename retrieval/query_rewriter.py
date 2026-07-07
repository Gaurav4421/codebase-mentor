"""
Multi-query expansion: asks the LLM for a handful of alternative search
queries that approach the question from different angles (synonyms,
implementation-level terms, related concepts), so retrieval doesn't live or
die on the user's exact phrasing. Each generated query is run through hybrid
retrieval independently and the results are merged (see adaptive_retriever.py).
"""
import logging
from typing import List

from llm.models import LLMClient
from retrieval.query_classifier import QueryIntent

logger = logging.getLogger(__name__)

DEFAULT_NUM_REWRITES = 3

_SKIP_REWRITE_INTENTS = (QueryIntent.FILE_SEARCH,)


def generate_search_queries(
    query: str,
    llm: LLMClient,
    intent: QueryIntent,
    num_rewrites: int = DEFAULT_NUM_REWRITES,
) -> List[str]:
    """Returns [original_query, *rewrites]. Falls back to just the original
    query if rewriting is skipped or the LLM call/parsing fails -- retrieval
    quality degrading to "single query" beats crashing the turn."""
    if intent in _SKIP_REWRITE_INTENTS:
        return [query]

    prompt = f"""A user is asking a question about a codebase:

"{query}"

Generate {num_rewrites} alternative search queries that would help find the relevant code, each \
approaching the question from a different angle (implementation-level terminology, related \
concepts, likely function/class/file names). Do not just rephrase the question conversationally -- \
think about what a keyword/semantic search over source code would need.

Respond with ONLY the {num_rewrites} queries, one per line, no numbering, no commentary."""

    try:
        raw = llm.generate(prompt)
    except Exception:
        logger.exception("Query rewriting LLM call failed; falling back to the original query")
        return [query]

    rewrites = [line.strip("-•* ").strip() for line in raw.splitlines()]
    rewrites = [r for r in rewrites if r and r.lower() != query.lower()]

    return [query] + rewrites[:num_rewrites]
