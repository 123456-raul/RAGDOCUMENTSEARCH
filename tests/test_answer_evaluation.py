"""Tests for the answer evaluation's judging code, with stand-ins for the models"""

from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from evaluation.evaluate_answers import evaluate_question, parse_verdict, with_retries


@pytest.mark.parametrize(
    "reply, passed, reason",
    [
        ("yes - matches the reference", True, "matches the reference"),
        ("No — the answer says 6, not 8.", False, "the answer says 6, not 8."),
        ("**Yes**: same fact", True, "same fact"),
        ("YES", True, ""),
    ],
)
def test_parse_verdict(reply, passed, reason):
    assert parse_verdict(reply) == (passed, reason)


def test_unclear_verdict_counts_as_fail():
    passed, reason = parse_verdict("It depends on the reading.")
    assert passed is False
    assert reason.startswith("unclear verdict")


def test_a_word_starting_with_no_is_not_a_no():
    assert parse_verdict("Nothing here is wrong")[1].startswith("unclear verdict")


class FakeJudge:
    """Says yes to the correctness question and no to the grounding question"""

    def __init__(self):
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        reply = "no - not in the passages" if "Passages:" in prompt else "yes - same fact"
        return SimpleNamespace(content=reply)


class FakeApp:
    def run(self, question):
        return {
            "answer": "It uses 8 heads [1].",
            "retrieved_docs": [Document(page_content="we employ h = 8 heads")],
            "rewrites": 1,
        }


def test_evaluate_question_judges_correctness_and_grounding_separately():
    judge = FakeJudge()
    item = {"question": "How many heads?", "answer": "8."}

    result = evaluate_question(FakeApp(), judge, item)

    assert result["correct"] is True
    assert result["grounded"] is False
    assert result["cited"] is True
    assert result["rewrites"] == 1
    # The correctness judge sees the reference; the grounding judge sees the passages
    assert "Reference answer: 8." in judge.prompts[0]
    assert "we employ h = 8 heads" in judge.prompts[1]


def test_with_retries_waits_out_a_rate_limit():
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("Error code: 429 - rate limit reached")
        return "done"

    assert with_retries(flaky, wait=0) == "done"
    assert len(calls) == 3


def test_with_retries_does_not_hide_other_errors():
    def broken():
        raise ValueError("bad key")

    with pytest.raises(ValueError):
        with_retries(broken, wait=0)
