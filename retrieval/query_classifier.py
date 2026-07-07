"""
Classifies a query into a retrieval strategy, not a prompt template. This is
deliberately separate from llm.prompts.classify_intent: that function picks
which *prompt* to render; this one picks which *retrieval path* to run
(adaptive_retriever.py). They often agree, but not always -- e.g. a
"dependency question" and an "explanation" both render the general
explanation prompt, but need very different retrieval.

Kept as a keyword classifier, matching the rest of the codebase's philosophy
(prompts.classify_intent): zero latency/cost, easy to read and extend.
"""
from enum import Enum


class QueryIntent(str, Enum):
    CODE_EXPLANATION = "code_explanation"
    BUG_DEBUGGING = "bug_debugging"
    ARCHITECTURE = "architecture_understanding"
    FILE_SEARCH = "file_search"
    DEPENDENCY_QUESTION = "dependency_question"
    DOCUMENTATION_GENERATION = "documentation_generation"
    GENERAL_QA = "general_qa"


_FILE_SEARCH_KEYWORDS = (
    "which file", "what file", "where is", "where's", "find the file",
    "locate", "what folder", "which folder", "file for", "file that",
)
_DEPENDENCY_KEYWORDS = (
    "depends on", "depend on", "dependency", "dependencies", "imports", "imported by",
    "who calls", "callers of", "called by", "used by", "uses ", "calls ",
    "inherits", "subclass", "extends", "call graph", "which functions call",
)
_DEBUGGING_KEYWORDS = (
    "bug", "error", "exception", "fail", "crash", "traceback", "not working",
    "doesn't work", "wrong output", "fix ", "why is", "why does", "broken",
)
_ARCHITECTURE_KEYWORDS = (
    "architecture", "design", "how does the system", "overall structure",
    "high level", "high-level", "module", "components", "flow", "pipeline",
    "how do the pieces fit", "system design",
)
_DOCUMENTATION_KEYWORDS = (
    "write docs", "generate documentation", "document this", "docstring",
    "readme for", "write a readme", "api docs", "documentation for",
)
_EXPLANATION_KEYWORDS = (
    "explain", "what does", "what is", "how does", "walk me through", "understand",
)


def classify_query(query: str) -> QueryIntent:
    """Order matters: more specific categories (file search, dependency,
    debugging, documentation) are checked before the broader
    explanation/architecture buckets they'd otherwise be swallowed by."""
    q = query.lower()

    if any(k in q for k in _FILE_SEARCH_KEYWORDS):
        return QueryIntent.FILE_SEARCH
    if any(k in q for k in _DEPENDENCY_KEYWORDS):
        return QueryIntent.DEPENDENCY_QUESTION
    if any(k in q for k in _DEBUGGING_KEYWORDS):
        return QueryIntent.BUG_DEBUGGING
    if any(k in q for k in _DOCUMENTATION_KEYWORDS):
        return QueryIntent.DOCUMENTATION_GENERATION
    if any(k in q for k in _ARCHITECTURE_KEYWORDS):
        return QueryIntent.ARCHITECTURE
    if any(k in q for k in _EXPLANATION_KEYWORDS):
        return QueryIntent.CODE_EXPLANATION
    return QueryIntent.GENERAL_QA
