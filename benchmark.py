"""
A small, real evaluation harness -- not a placeholder. Runs a set of
(question, expected-signal) cases against a live CodebaseMentor index and
scores two things that matter independently:

1. Retrieval hit-rate: did the retrieved chunks include at least one of the
   expected files/symbols? (checks retrieval quality in isolation)
2. Answer grounding: does the final answer text mention the expected
   files/symbols? (checks the whole pipeline, including the LLM call)

This intentionally does NOT try to score "answer correctness" via another LLM
call (LLM-graded eval is a reasonable next step, but it hides failures behind
another model's judgment -- string/metadata matching against ground truth you
picked yourself is more trustworthy for a first benchmark).
"""
import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parent.parent))

load_dotenv()

from llm.models import GeminiClient
from pipeline import CodebaseMentor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkCase:
    question: str
    # any of these appearing in a retrieved chunk's path, or in the answer text,
    # counts as a hit for that case
    expected_signals: List[str] = field(default_factory=list)


@dataclass
class CaseResult:
    case: BenchmarkCase
    intent: str
    retrieval_hit: bool
    answer_grounded: bool
    answer: str


def run_benchmark(mentor: CodebaseMentor, cases: List[BenchmarkCase]) -> List[CaseResult]:
    memory = mentor.new_conversation()
    results = []

    for case in cases:
        answer, intent, retrieved_docs = mentor.ask(case.question, memory)
        memory.add_user_turn(case.question)
        memory.add_assistant_turn(answer)

        retrieved_paths = " ".join(d.metadata.get("path", "") for d in retrieved_docs)
        retrieval_hit = any(signal.lower() in retrieved_paths.lower() for signal in case.expected_signals)
        answer_grounded = any(signal.lower() in answer.lower() for signal in case.expected_signals)

        results.append(CaseResult(
            case=case, intent=intent, retrieval_hit=retrieval_hit,
            answer_grounded=answer_grounded, answer=answer,
        ))

    return results


def summarize(results: List[CaseResult]) -> None:
    total = len(results)
    retrieval_hits = sum(r.retrieval_hit for r in results)
    grounded = sum(r.answer_grounded for r in results)

    print(f"\n{'=' * 60}")
    print(f"Cases run:            {total}")
    print(f"Retrieval hit-rate:   {retrieval_hits}/{total} ({retrieval_hits / total:.0%})")
    print(f"Answer grounding:     {grounded}/{total} ({grounded / total:.0%})")
    print(f"{'=' * 60}\n")

    for r in results:
        status = "OK" if r.retrieval_hit and r.answer_grounded else "CHECK"
        print(f"[{status}] ({r.intent}) {r.case.question}")
        if not r.retrieval_hit:
            print(f"         retrieval missed expected signals: {r.case.expected_signals}")
        if not r.answer_grounded:
            print(f"         answer did not mention expected signals: {r.case.expected_signals}")


def default_cases() -> List[BenchmarkCase]:
    """Generic, repo-agnostic starter cases. Real usage should replace these
    with cases tailored to the repo under test (specific file/symbol names)."""
    return [
        BenchmarkCase(
            question="What is this repository and what does it do?",
            expected_signals=[],  # repo_summary intent has no single "correct" file
        ),
        BenchmarkCase(
            question="What are the main modules or components in this codebase?",
            expected_signals=[],
        ),
    ]


def main():
    parser = argparse.ArgumentParser(description="Run the codebase-mentor benchmark against a repo.")
    parser.add_argument("repo_path", help="Path to the repository to index and query.")
    parser.add_argument("--json-out", help="Optional path to write results as JSON.", default=None)
    args = parser.parse_args()

    mentor = CodebaseMentor(llm=GeminiClient())
    mentor.index_repository(args.repo_path)

    results = run_benchmark(mentor, default_cases())
    summarize(results)

    if args.json_out:
        payload = [
            {
                "question": r.case.question,
                "expected_signals": r.case.expected_signals,
                "intent": r.intent,
                "retrieval_hit": r.retrieval_hit,
                "answer_grounded": r.answer_grounded,
                "answer": r.answer,
            }
            for r in results
        ]
        Path(args.json_out).write_text(json.dumps(payload, indent=2))
        logger.info("Wrote benchmark results to %s", args.json_out)


if __name__ == "__main__":
    main()
