"""
Everything about *shaping* what gets sent to the LLM: which prompt template to
use (classify_intent), what context to put in it (build_context), and how to
render the guaranteed-non-hallucinated source list appended to every answer
(build_sources_footer).

Direct port of the notebook's prompting section -- behavior unchanged, just
without the `client`/`GEMINI_MODEL` globals (callers pass an LLMClient in via
llm/models.py, used elsewhere in the pipeline, not in this module directly).
"""
import os
from pathlib import Path
from typing import Dict, List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

_GROUNDING_RULES = """Ground everything in the CONTEXT below -- if something isn't covered by it, say so \
plainly instead of guessing. Reference specific files/functions/classes by name when relevant \
(e.g. "in `foo.py`'s `Bar.baz` method..."). Use bullet points only for genuinely list-like information; \
use prose for explanations. Be concise but complete -- don't pad, and don't reduce a real explanation \
down to a stub sentence. You are in the middle of an ongoing conversation -- use the chat history to \
resolve references like "it", "that function", "there", etc., and don't repeat information you already \
gave earlier unless the user is asking for it again."""


PROMPT_TEMPLATES: Dict[str, ChatPromptTemplate] = {
    "repository_summary": ChatPromptTemplate.from_messages([
        ("system",
         "You are a senior engineer giving a newcomer a guided tour of this codebase. Answer using the "
         "Repository Overview and Relevant Module(s) sections of the CONTEXT -- repo type, languages/"
         "frameworks in use, main components, entry points, and major dependencies. Only drop into "
         "specific retrieved code if the question asks for it. " + _GROUNDING_RULES),
        MessagesPlaceholder("chat_history"),
        ("human", "CONTEXT:\n{context}\n\nQUESTION:\n{question}"),
    ]),
    "explanation": ChatPromptTemplate.from_messages([
        ("system",
         "You are a senior engineer walking a teammate through this codebase in a design discussion. "
         "Explain *what* the relevant code does and *why* it's built that way. " + _GROUNDING_RULES),
        MessagesPlaceholder("chat_history"),
        ("human", "CONTEXT:\n{context}\n\nQUESTION:\n{question}"),
    ]),
    "debugging": ChatPromptTemplate.from_messages([
        ("system",
         "You are a senior engineer helping debug an issue in this codebase. Walk through the likely "
         "root cause step by step using the CONTEXT: what the code currently does, where it could go "
         "wrong or diverge from the expected behavior, and what you'd check or change next. "
         + _GROUNDING_RULES),
        MessagesPlaceholder("chat_history"),
        ("human", "CONTEXT:\n{context}\n\nISSUE / QUESTION:\n{question}"),
    ]),
    "architecture": ChatPromptTemplate.from_messages([
        ("system",
         "You are a senior engineer documenting this codebase's architecture. Use the repository summary "
         "and module summaries in the CONTEXT to describe how the relevant pieces fit together -- "
         "responsibilities, boundaries, and how data/control flows between them -- before drilling into "
         "specific code. " + _GROUNDING_RULES),
        MessagesPlaceholder("chat_history"),
        ("human", "CONTEXT:\n{context}\n\nQUESTION:\n{question}"),
    ]),
    "comparison": ChatPromptTemplate.from_messages([
        ("system",
         "You are a senior engineer comparing two or more things (functions, classes, approaches, files) "
         "found in this codebase. Structure your answer around their similarities, differences, and "
         "trade-offs, backed by the CONTEXT. " + _GROUNDING_RULES),
        MessagesPlaceholder("chat_history"),
        ("human", "CONTEXT:\n{context}\n\nQUESTION:\n{question}"),
    ]),
    "general_qa": ChatPromptTemplate.from_messages([
        ("system",
         "You are a senior engineer answering a question about this codebase in a code review. "
         + _GROUNDING_RULES),
        MessagesPlaceholder("chat_history"),
        ("human", "CONTEXT:\n{context}\n\nQUESTION:\n{question}"),
    ]),
}

_REPO_SUMMARY_KEYWORDS = (
    "what is this repo", "what does this repo", "what does this project", "overview of",
    "summarize the repo", "summarise the repo", "tell me about this repo",
    "tell me about this project", "what tech stack", "what frameworks", "what languages", "readme",
)
_DEBUGGING_KEYWORDS = (
    "bug", "error", "exception", "fail", "crash", "traceback", "not working",
    "doesn't work", "wrong output", "fix ", "why is", "why does", "broken",
)
_ARCHITECTURE_KEYWORDS = (
    "architecture", "design", "how does the system", "overall structure",
    "high level", "high-level", "module", "components", "flow", "pipeline",
)
_COMPARISON_KEYWORDS = (
    " vs ", " versus ", "difference between", "compare", "which is better",
    "pros and cons", "trade-off", "tradeoff",
)
_EXPLANATION_KEYWORDS = (
    "explain", "what does", "what is", "how does", "walk me through", "understand",
)


def classify_intent(query: str) -> str:
    """Lightweight keyword-based intent classifier. Falls back to general_qa
    when nothing matches. Kept intentionally simple (no ML classifier) -- it's
    cheap, has zero latency/cost, and is easy to extend/debug by reading the
    keyword lists above."""
    q = query.lower()

    if any(k in q for k in _REPO_SUMMARY_KEYWORDS):
        return "repository_summary"
    if any(k in q for k in _DEBUGGING_KEYWORDS):
        return "debugging"
    if any(k in q for k in _COMPARISON_KEYWORDS):
        return "comparison"
    if any(k in q for k in _ARCHITECTURE_KEYWORDS):
        return "architecture"
    if any(k in q for k in _EXPLANATION_KEYWORDS):
        return "explanation"
    return "general_qa"


def _module_of(path: str, repo_root: str) -> str:
    rel = os.path.relpath(path, repo_root)
    parts = Path(rel).parts
    return parts[0] if len(parts) > 1 else "(root)"


def build_context(
    results: List[Document],
    repo_summary: str,
    module_summaries: dict,
    repo_path: str,
) -> str:
    """Layers three things instead of just dumping raw chunks at the model:
    1. repository summary (always included -- cheap, global orientation)
    2. module summary/summaries relevant to whatever files were retrieved
    3. the retrieved code chunks themselves, labeled with file/class/name
    """
    sections = [f"## Repository Overview\n{repo_summary}"]

    touched_modules = {}
    for doc in results:
        mod = _module_of(doc.metadata["path"], repo_path)
        if mod in module_summaries and mod not in touched_modules:
            summary = module_summaries[mod]
            touched_modules[mod] = summary.summary if hasattr(summary, "summary") else summary.get("summary", "")

    if touched_modules:
        module_lines = [f"- **{name}**: {summary}" for name, summary in touched_modules.items()]
        sections.append("## Relevant Module(s)\n" + "\n".join(module_lines))

    context_blocks = []
    for doc in results:
        loc = f"{doc.metadata['path']}:{doc.metadata['start_line']}-{doc.metadata['end_line']}"
        label = doc.metadata.get("name") or doc.metadata.get("type", "code")
        if doc.metadata.get("parent_class"):
            label = f"{doc.metadata['parent_class']}.{label}"
        lang = doc.metadata.get("language", "")
        context_blocks.append(f"### {label}  ({loc})\n```{lang}\n{doc.page_content}\n```")

    sections.append("## Retrieved Code\n" + "\n\n".join(context_blocks))
    return "\n\n".join(sections)


def build_sources_footer(results: List[Document]) -> str:
    """Deterministic, non-hallucinated source list built directly from
    retrieved-chunk metadata -- never generated by the LLM, so it can't invent
    files/functions that were never retrieved."""
    if not results:
        return "Sources: none retrieved -- answer is not grounded in this repository."

    seen = set()
    lines = []
    for doc in results:
        meta = doc.metadata
        loc = f"{meta['path']}:{meta['start_line']}-{meta['end_line']}"
        if loc in seen:
            continue
        seen.add(loc)

        label = meta.get("name") or meta.get("type", "code")
        if meta.get("parent_class"):
            label = f"{meta['parent_class']}.{label}"
        lines.append(f"- `{label}` -- {loc}")

    return "Sources:\n" + "\n".join(lines)
