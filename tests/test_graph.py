"""Tests for the LangGraph workflow: the relevance branch, the rewrite loop, chat memory"""

from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from src.config.config import Config
from src.graph_builder.graph_builder import GraphBuilder
from src.node.reactnode import format_passages, message_text


class FakeRetriever:
    """Returns one passage and remembers what it was asked"""

    def __init__(self):
        self.queries = []

    def invoke(self, query):
        self.queries.append(query)
        return [Document(page_content=f"passage for {query}", metadata={"source": "a.txt"})]


class FakeLLM:
    """Plays the grader, the rewriter and the follow-up rewriter"""

    def __init__(self, verdicts):
        self.verdicts = list(verdicts)  # the grader's replies, in order
        self.prompts = []

    def invoke(self, prompt, config=None):
        self.prompts.append(prompt)
        if "checking search results" in prompt:
            reply = self.verdicts.pop(0)
        elif "Write a better search query" in prompt:
            reply = "better query"
        else:
            reply = "standalone question"
        return SimpleNamespace(content=reply)


def build(verdicts):
    retriever, llm = FakeRetriever(), FakeLLM(verdicts)
    builder = GraphBuilder(retriever, llm)
    # Replace the agent, which would need a real model, with a canned answer
    builder.nodes.generate_answer = lambda state: {"answer": "final answer"}
    builder.build()
    return builder, retriever, llm


@pytest.fixture(autouse=True)
def settings(monkeypatch):
    monkeypatch.setattr(Config, "CHECK_RELEVANCE", True)
    monkeypatch.setattr(Config, "MAX_REWRITES", 1)


def test_relevant_passages_go_straight_to_the_answer():
    builder, retriever, _ = build(["yes"])
    result = builder.run("What is attention?")

    assert retriever.queries == ["What is attention?"]
    assert result["rewrites"] == 0
    assert result["docs_relevant"] is True
    assert result["answer"] == "final answer"


def test_irrelevant_passages_trigger_a_rewrite_and_a_second_search():
    builder, retriever, _ = build(["no", "yes"])
    result = builder.run("What is attention?")

    assert retriever.queries == ["What is attention?", "better query"]
    assert result["rewrites"] == 1
    assert result["search_query"] == "better query"
    assert result["docs_relevant"] is True


def test_the_loop_stops_after_the_allowed_rewrites():
    builder, retriever, llm = build(["no", "no", "no"])
    result = builder.run("What is attention?")

    assert len(retriever.queries) == 2  # the first search and one retry
    assert result["rewrites"] == 1
    assert result["docs_relevant"] is False
    assert result["answer"] == "final answer"
    assert llm.verdicts == ["no"]  # the third verdict was never asked for


def test_relevance_check_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(Config, "CHECK_RELEVANCE", False)
    builder, retriever, llm = build([])
    result = builder.run("What is attention?")

    assert llm.prompts == []
    assert len(retriever.queries) == 1
    assert result["docs_relevant"] is True


def test_follow_up_question_is_rewritten_using_the_conversation():
    builder, retriever, llm = build(["yes"])
    history = [
        {"role": "user", "content": "What is attention?"},
        {"role": "assistant", "content": "A way to weigh tokens."},
    ]
    result = builder.run("Explain that more", history)

    assert "What is attention?" in llm.prompts[0]
    assert retriever.queries == ["standalone question"]
    assert result["question"] == "Explain that more"  # the answer still addresses the user's words


def test_stream_reports_steps_in_order_and_ends_with_the_result():
    builder, _, _ = build(["no", "yes"])
    events = list(builder.stream("What is attention?"))

    steps = [payload for kind, payload in events if kind == "step"]
    assert steps == [
        "contextualize", "retriever", "grader", "rewriter", "retriever", "grader", "responder",
    ]
    assert events[-1][0] == "result"
    assert events[-1][1]["answer"] == "final answer"


def test_message_text_drops_thinking_and_joins_parts():
    assert message_text(SimpleNamespace(content="<think>hmm</think> Yes")) == "Yes"
    assert message_text(SimpleNamespace(content=[{"type": "text", "text": "a"}, "b"])) == "ab"


def test_passages_are_numbered_with_their_source():
    docs = [Document(page_content="text", metadata={"source": "data/paper.pdf", "page": 0})]
    assert format_passages(docs) == "[1] (paper.pdf, page 1)\ntext"
