"""
Conversation state as real LangChain message objects, follow-up-question
resolution, and history trimming/summarization. Direct port of the notebook's
conversation-memory section, wrapped in a class and taking an injected
LLMClient instead of reaching for a global `client`.
"""
import logging
from typing import List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from llm.models import LLMClient

logger = logging.getLogger(__name__)

DEFAULT_MAX_TURNS = 6  


def _flatten_for_llm(messages: List[BaseMessage]) -> str:
    """google-genai's generate_content wants a single string, so role-label
    each LangChain message instead of concatenating raw text blindly (keeps
    Human/AI turns distinguishable to the model)."""
    return "\n\n".join(f"[{m.type.upper()}]\n{m.content}" for m in messages)


class ConversationMemory:
    """One instance per active conversation (one per Streamlit session, one
    per benchmark run, etc.) -- state lives on the instance, not in module
    globals, so multiple conversations can run concurrently."""

    def __init__(self, llm: LLMClient, max_turns: int = DEFAULT_MAX_TURNS):
        self.llm = llm
        self.max_turns = max_turns
        self.history: List[BaseMessage] = []

    def add_user_turn(self, query: str) -> None:
        self.history.append(HumanMessage(content=query))

    def add_assistant_turn(self, answer: str) -> None:
        self.history.append(AIMessage(content=answer))

    def resolve_followup_query(self, query: str) -> str:
        """Rewrite `query` into a standalone question if -- and only if -- it
        depends on earlier turns (pronouns like "it"/"that"/"there", or an
        implicit subject). Returns the query unchanged if it already stands on
        its own, or there's no history yet. This keeps retrieval quality high
        on follow-ups without the retrieval pipeline itself ever changing."""
        non_system_history = [m for m in self.history if not isinstance(m, SystemMessage)]
        if not non_system_history:
            return query

        recent = non_system_history[-(self.max_turns * 2):]
        history_text = _flatten_for_llm(recent)

        rewrite_prompt = f"""Conversation so far:
{history_text}

New user message: "{query}"

If the new user message depends on the conversation above to make sense (e.g. it uses "it", "that",
"this", "there", or otherwise assumes context from earlier turns), rewrite it as a fully standalone
question that a search engine could answer with no prior context. If it is already standalone, return
it completely unchanged.

Respond with ONLY the resulting question -- no quotes, no explanation, no preamble."""

        try:
            rewritten = self.llm.generate(rewrite_prompt).strip().strip('"')
            return rewritten if rewritten else query
        except Exception:
            # Retrieval degrading gracefully to "use the raw query" beats
            # crashing the whole turn over a rewrite hiccup.
            logger.exception("resolve_followup_query LLM call failed; using raw query")
            return query

    def _summarize_messages(self, messages: List[BaseMessage]) -> str:
        text = _flatten_for_llm(messages)
        prompt = f"""Summarize the following conversation between a user and a codebase assistant.
Preserve concrete facts: file names, function/class names, and any conclusions already given to the
user -- a later turn may still refer back to them. Keep it under 6 sentences.

CONVERSATION:
{text}

Respond with ONLY the summary."""
        try:
            return self.llm.generate(prompt).strip()
        except Exception:
            logger.exception("History summarization LLM call failed")
            return "(earlier conversation history -- summary unavailable)"

    def trim(self) -> None:
        """Keep the last `max_turns` human/AI pairs verbatim; collapse anything
        older (plus any existing running summary) into a single leading
        SystemMessage. Mutates self.history in place."""
        leading_summary = (
            self.history[0] if self.history and isinstance(self.history[0], SystemMessage) else None
        )
        rest = self.history[1:] if leading_summary else self.history

        keep_count = self.max_turns * 2
        if len(rest) <= keep_count:
            return  # nothing to trim yet

        to_summarize = rest[:-keep_count]
        to_keep = rest[-keep_count:]
        if leading_summary is not None:
            to_summarize = [leading_summary] + to_summarize

        summary_text = self._summarize_messages(to_summarize)
        self.history = [SystemMessage(content=f"Summary of earlier conversation: {summary_text}")] + to_keep

    def as_prompt_history(self) -> List[BaseMessage]:
        """What gets handed to MessagesPlaceholder('chat_history') -- does not
        include the current turn's message, which the caller adds afterward."""
        return self.history
