"""
Groups symbols into modules (top-level folders) and produces the repository-
and module-level natural-language summaries used by the `repository_summary`
and `architecture` prompt templates.

This is a direct port of the notebook's `generate_repository_intelligence`,
split into a deterministic part (module grouping, always correct, no LLM) and
an LLM-grounded part (summarization). The LLM is injected as an `LLMClient`
(see llm/models.py) instead of a hardcoded Gemini call, so this is unit
testable with FakeLLMClient.
"""
import json
import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from code_intelligence.symbol_extractor import Symbol, format_symbol_label
from llm.models import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class ModuleSummary:
    files: List[str] = field(default_factory=list)
    summary: str = ""


@dataclass
class RepositoryIntelligence:
    repo_summary: str
    modules: Dict[str, ModuleSummary]
    symbols_by_file: Dict[str, List[Symbol]]


def _module_of(path: str, repo_root: str) -> str:
    """First path component under the repo root = the 'module' for grouping."""
    rel = os.path.relpath(path, repo_root)
    parts = Path(rel).parts
    return parts[0] if len(parts) > 1 else "(root)"


def group_into_modules(symbols_by_file: Dict[str, List[Symbol]], repo_path: str) -> Dict[str, List[str]]:
    modules: Dict[str, set] = defaultdict(set)
    for path in symbols_by_file:
        modules[_module_of(path, repo_path)].add(path)
    return {name: sorted(files) for name, files in modules.items()}


def _safe_json_parse(text: str):
    """Gemini sometimes wraps JSON in ```json fences -- strip those before
    parsing. Returns None on failure rather than raising, so the caller can
    fall back to a heuristic summary instead of aborting indexing."""
    cleaned = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return None


def _build_evidence_text(modules: Dict[str, List[str]], symbols_by_file: Dict[str, List[Symbol]]) -> str:
    lines = []
    for module_name, files in sorted(modules.items()):
        lines.append(f"MODULE: {module_name}")
        for f in files[:15]:
            names = [format_symbol_label(s) for s in symbols_by_file.get(f, [])][:12]
            lines.append(f"  - {f}: {', '.join(names) if names else '(no named symbols)'}")
    return "\n".join(lines)


def generate_repository_intelligence(
    repo_path: str,
    symbols_by_file: Dict[str, List[Symbol]],
    llm: LLMClient,
) -> RepositoryIntelligence:
    """One-time, indexing-time pass: groups files into modules, then makes a
    single LLM call to summarize the whole repo + every module in one shot
    (cheap: one call regardless of repo size, grounded in real symbol names
    so it can't invent functionality)."""
    modules = group_into_modules(symbols_by_file, repo_path)
    evidence = _build_evidence_text(modules, symbols_by_file)

    prompt = f"""You are documenting a codebase for a new engineer joining the project.
Below is a list of modules (top-level folders), the files in each, and the functions/classes/methods
found in each file (extracted directly from the source, so treat these names as ground truth).

{evidence}

Respond with ONLY a JSON object (no markdown fences, no commentary) of the form:
{{
  "repo_summary": "2-4 sentence plain-English summary of what this repository does as a whole",
  "modules": {{
    "<module_name>": "1-2 sentence summary of this module's responsibility, grounded in its files/capabilities"
  }}
}}"""

    raw_response = llm.generate(prompt)
    parsed = _safe_json_parse(raw_response)

    if parsed and "repo_summary" in parsed:
        repo_summary = parsed.get("repo_summary", "").strip()
        module_text = parsed.get("modules", {})
    else:
        logger.warning("LLM repo-summary response was not valid JSON; falling back to heuristic summary")
        repo_summary = (
            f"Repository at {repo_path} contains {len(symbols_by_file)} source files "
            f"across {len(modules)} top-level module(s): {', '.join(sorted(modules))}."
        )
        module_text = {name: f"Contains {len(files)} file(s)." for name, files in modules.items()}

    module_summaries = {
        name: ModuleSummary(files=files, summary=module_text.get(name, ""))
        for name, files in modules.items()
    }

    return RepositoryIntelligence(
        repo_summary=repo_summary,
        modules=module_summaries,
        symbols_by_file=symbols_by_file,
    )
