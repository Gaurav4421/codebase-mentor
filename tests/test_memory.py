from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from llm.models import FakeLLMClient
from memory.conversation import ConversationMemory


def test_resolve_followup_query_returns_unchanged_with_no_history():
    llm = FakeLLMClient(response="should never be used")
    memory = ConversationMemory(llm)

    result = memory.resolve_followup_query("What does the parser do?")

    assert result == "What does the parser do?"
    assert llm.calls == []  # no history yet -> short-circuits before calling the LLM


def test_resolve_followup_query_rewrites_with_history():
    llm = FakeLLMClient(response="What does the Calculator class do?")
    memory = ConversationMemory(llm)
    memory.add_user_turn("Tell me about the Calculator class")
    memory.add_assistant_turn("It's a simple arithmetic helper.")

    result = memory.resolve_followup_query("What methods does it have?")

    assert result == "What does the Calculator class do?"
    assert len(llm.calls) == 1


def test_resolve_followup_query_falls_back_to_raw_query_on_llm_error():
    def raise_error(_prompt):
        raise RuntimeError("network down")

    llm = FakeLLMClient(responder=raise_error)
    memory = ConversationMemory(llm)
    memory.add_user_turn("first turn")
    memory.add_assistant_turn("first answer")

    result = memory.resolve_followup_query("and that?")

    assert result == "and that?"  # degrades gracefully instead of raising


def test_add_turns_appends_correct_message_types():
    memory = ConversationMemory(FakeLLMClient())
    memory.add_user_turn("hello")
    memory.add_assistant_turn("hi there")

    assert isinstance(memory.history[0], HumanMessage)
    assert isinstance(memory.history[1], AIMessage)
    assert memory.history[0].content == "hello"
    assert memory.history[1].content == "hi there"


def test_trim_keeps_recent_turns_verbatim_below_threshold():
    memory = ConversationMemory(FakeLLMClient(response="summary"), max_turns=2)
    for i in range(2):
        memory.add_user_turn(f"q{i}")
        memory.add_assistant_turn(f"a{i}")

    memory.trim()

    # exactly max_turns pairs -> nothing to summarize yet
    assert len(memory.history) == 4
    assert not isinstance(memory.history[0], SystemMessage)


def test_trim_summarizes_older_turns_once_over_threshold():
    llm = FakeLLMClient(response="condensed summary")
    memory = ConversationMemory(llm, max_turns=2)
    for i in range(4):  # 4 pairs, threshold is 2 -> 2 oldest pairs should collapse
        memory.add_user_turn(f"q{i}")
        memory.add_assistant_turn(f"a{i}")

    memory.trim()

    assert isinstance(memory.history[0], SystemMessage)
    assert "condensed summary" in memory.history[0].content
    # the 2 most recent pairs (4 messages) remain verbatim after the summary
    assert len(memory.history) == 1 + 4
    assert memory.history[-1].content == "a3"
