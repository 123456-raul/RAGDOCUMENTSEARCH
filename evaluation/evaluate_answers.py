"""Measure the quality of the final answers, not just the search.

evaluate.py checks that the right passage is found. This script runs every
question through the whole workflow and has a model judge the answer twice:

- Correct: does the answer state the same fact as the reference answer in
  questions.json?
- Grounded: is every claim in the answer supported by the passages the
  workflow retrieved? An answer can be correct but not grounded (the model
  used its own knowledge) or grounded but not correct (it picked the wrong
  fact from the passages).

The judge is the same model that writes the answers (LLM_PROVIDER in .env),
so read the verdicts in answer_results.md rather than trusting the totals.

Run from the project root:
    uv run python evaluation/evaluate_answers.py
"""

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.graph_builder.graph_builder import GraphBuilder
from src.node.reactnode import format_passages, message_text
from src.vectorstore.vectorstore import VectorStore

QUESTIONS_FILE = Path(__file__).with_name("questions.json")
RESULTS_FILE = Path(__file__).with_name("answer_results.md")

RETRIES = 5  # for rate limits on free API tiers
CITATION = re.compile(r"\[\d+\]")

CORRECT_PROMPT = (
    "You are grading an answer to a question about a document.\n\n"
    "Question: {question}\n"
    "Reference answer: {reference}\n"
    "Answer to grade: {answer}\n\n"
    "Does the answer to grade state the same fact as the reference answer? "
    "Extra correct detail is fine. A different fact, a missing fact, or "
    "\"I don't know\" is not.\n"
    "Reply with one word, yes or no, then a dash and a reason of at most 15 words."
)

GROUNDED_PROMPT = (
    "You are checking whether an answer is supported by source passages.\n\n"
    "Passages:\n{passages}\n\n"
    "Answer to check: {answer}\n\n"
    "Is every factual claim in the answer stated in the passages? Ignore "
    "citation numbers and wording; judge only the facts.\n"
    "Reply with one word, yes or no, then a dash and a reason of at most 15 words."
)


def parse_verdict(reply: str):
    """Split a judge's reply into (passed, reason); an unclear reply is a fail"""
    text = reply.strip()
    match = re.match(r"\W*(yes|no)\b[\s\-–—:.,*]*(.*)", text, re.IGNORECASE | re.DOTALL)
    if not match:
        return False, f"unclear verdict: {text[:80]}"
    return match.group(1).lower() == "yes", " ".join(match.group(2).split())


def with_retries(call, retries: int = RETRIES, wait: float = 20.0):
    """Run `call`, waiting and retrying when the provider says to slow down"""
    for attempt in range(retries):
        try:
            return call()
        except Exception as error:
            rate_limited = "429" in str(error) or "rate" in str(error).lower()
            if not rate_limited or attempt == retries - 1:
                raise
            time.sleep(wait * (attempt + 1))


def judge(llm, prompt: str):
    reply = with_retries(lambda: llm.invoke(prompt))
    return parse_verdict(message_text(reply))


def evaluate_question(app, llm, item) -> dict:
    """Answer one question with the workflow and judge the answer"""
    state = with_retries(lambda: app.run(item["question"]))
    answer = state["answer"]
    passages = state["retrieved_docs"]

    correct, correct_why = judge(
        llm,
        CORRECT_PROMPT.format(
            question=item["question"], reference=item["answer"], answer=answer
        ),
    )
    grounded, grounded_why = judge(
        llm,
        GROUNDED_PROMPT.format(passages=format_passages(passages), answer=answer),
    )
    return {
        "question": item["question"],
        "reference": item["answer"],
        "answer": answer,
        "correct": correct,
        "correct_why": correct_why,
        "grounded": grounded,
        "grounded_why": grounded_why,
        "cited": bool(CITATION.search(answer)),
        "rewrites": state["rewrites"],
    }


def write_report(results, model_name: str) -> None:
    n = len(results)
    share = lambda key: f"{sum(r[key] for r in results)} of {n}"
    mark = lambda passed: "yes" if passed else "**no**"
    cell = lambda text: " ".join(str(text).split()).replace("|", "\\|")

    lines = [
        "# Answer evaluation",
        "",
        f"{n} questions answered by the full workflow with `{model_name}` "
        f"({Config.RETRIEVER_TYPE} index, chunks of {Config.CHUNK_SIZE} / {Config.CHUNK_OVERLAP}). "
        "The same model judged the answers.",
        "",
        f"- **Correct** (same fact as the reference answer): {share('correct')}",
        f"- **Grounded** (every claim is in the retrieved passages): {share('grounded')}",
        f"- **Cited** (the answer points to a numbered passage): {share('cited')}",
        f"- **Searched twice** (the grader asked for a rewritten question): "
        f"{sum(r['rewrites'] > 0 for r in results)} of {n}",
        "",
        "| # | Question | Correct | Grounded | Cited |",
        "|---|---|---|---|---|",
    ]
    for i, r in enumerate(results, start=1):
        lines.append(
            f"| {i} | {cell(r['question'])} | {mark(r['correct'])} "
            f"| {mark(r['grounded'])} | {mark(r['cited'])} |"
        )

    lines += ["", "## Answers and the judge's reasons", ""]
    for i, r in enumerate(results, start=1):
        lines += [
            f"**{i}. {r['question']}**",
            "",
            f"- Reference: {cell(r['reference'])}",
            f"- Answer: {cell(r['answer'])}",
            f"- Correct: {mark(r['correct'])} — {cell(r['correct_why'])}",
            f"- Grounded: {mark(r['grounded'])} — {cell(r['grounded_why'])}",
            "",
        ]
    RESULTS_FILE.write_text("\n".join(lines), encoding="utf-8")


def main():
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))

    # The built-in sources only, so uploaded files don't change the numbers
    uploads = {str(path) for path in Config.get_uploads()}
    sources = [s for s in Config.get_sources() if s not in uploads]

    llm = Config.get_llm()
    store = VectorStore()
    store.load_or_build(sources, DocumentProcessor(Config.CHUNK_SIZE, Config.CHUNK_OVERLAP))
    app = GraphBuilder(retriever=store.get_retriever(), llm=llm)
    app.build()

    results = []
    for i, item in enumerate(questions, start=1):
        result = evaluate_question(app, llm, item)
        results.append(result)
        print(
            f"{i:2}/{len(questions)} correct={result['correct']!s:5} "
            f"grounded={result['grounded']!s:5} {item['question']}"
        )

    model_name = getattr(llm, "model_name", None) or getattr(llm, "model", Config.LLM_PROVIDER)
    write_report(results, model_name)
    print(f"\nSaved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
