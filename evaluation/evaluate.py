"""Measure how often retrieval finds the chunk that holds the answer.

Each question in questions.json comes with one or more phrases copied from the
source text. A search counts as a hit when one of the returned chunks contains
one of them. The script compares chunk sizes, the tree and flat index, and the
optional keyword-search and reranking steps, on the documents in data/.

Run from the project root:
    uv run python evaluation/evaluate.py
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
from src.vectorstore.vectorstore import VectorStore

QUESTIONS_FILE = Path(__file__).with_name("questions.json")
RESULTS_FILE = Path(__file__).with_name("results.md")

CHUNKINGS = [(500, 50), (1000, 150)]  # (chunk size, overlap) in characters
INDEXES = ["flat", "tree"]
# (label, keyword search, reranker)
PIPELINES = [
    ("embeddings only", False, False),
    ("+ keywords (BM25)", True, False),
    ("+ reranker", False, True),
    ("+ keywords + reranker", True, True),
]


def normalize(text: str) -> str:
    """Lowercase and collapse whitespace, so line breaks don't hide a match"""
    return re.sub(r"\s+", " ", text).lower()


def answer_phrases(item) -> list:
    """The accepted phrases for a question; "answer_in" is one phrase or a list.

    A list is for a fact the source states in more than one place in different
    words, such as "8 NVIDIA P100" and "eight P100 GPUs".
    """
    phrases = item["answer_in"]
    return [normalize(p) for p in ([phrases] if isinstance(phrases, str) else phrases)]


def holds_answer(text: str, phrases: list) -> bool:
    return any(phrase in text for phrase in phrases)


def evaluate(retriever, questions, chunk_texts, k):
    """Return hit rate, mean reciprocal rank, search time, share of chunks compared, and misses"""
    retriever.rank("warm up")  # load models before timing

    hits, reciprocal_ranks, seconds, compared = 0, 0.0, 0.0, 0.0
    missed = []
    for item in questions:
        start = time.perf_counter()
        ids = retriever.rank(item["question"], k=k)
        seconds += time.perf_counter() - start
        compared += retriever.last_search["compared"] / len(chunk_texts)

        phrases = answer_phrases(item)
        rank = next(
            (pos for pos, i in enumerate(ids, start=1) if holds_answer(chunk_texts[i], phrases)),
            None,
        )
        if rank is None:
            missed.append(item["question"])
        else:
            hits += 1
            reciprocal_ranks += 1.0 / rank

    n = len(questions)
    return hits / n, reciprocal_ranks / n, seconds / n * 1000, compared / n, missed


def main():
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    k = Config.RETRIEVER_K

    # The built-in sources only, so uploaded files don't change the numbers
    uploads = {str(path) for path in Config.get_uploads()}
    sources = [s for s in Config.get_sources() if s not in uploads]
    print(f"Loading {len(sources)} sources...")
    raw_docs = DocumentProcessor().load_documents(sources)
    store = VectorStore()

    lines = [
        "# Retrieval evaluation",
        "",
        f"{len(questions)} questions, top {k} chunks per question, "
        f"{Config.CANDIDATE_K} candidates before merging and reranking.",
        "",
        "- **Hit rate**: share of questions where a returned chunk contains the answer.",
        "- **MRR**: mean of 1 / (position of the first such chunk); 1.0 means always first.",
        "- **Compared**: share of chunk embeddings the query was compared with.",
        "- **Search**: average time per question on the machine that ran this.",
        "",
        "| Chunks (size / overlap, count) | Index | Pipeline | Hit rate | MRR | Compared | Search |",
        "|---|---|---|---|---|---|---|",
    ]
    missed_report = []

    for chunk_size, overlap in CHUNKINGS:
        chunks = DocumentProcessor(chunk_size, overlap).split_documents(raw_docs)
        chunk_texts = [normalize(c.page_content) for c in chunks]
        vectors = store.embed(chunks)

        # A phrase cut in half by a chunk boundary could never be found
        for item in questions:
            if not any(holds_answer(t, answer_phrases(item)) for t in chunk_texts):
                print(f"  WARNING no chunk holds the answer at size {chunk_size}: {item['question']}")

        for index_type in INDEXES:
            for label, hybrid, rerank in PIPELINES:
                retriever = store.build_retriever(
                    chunks, vectors, retriever_type=index_type, hybrid=hybrid, rerank=rerank
                )
                hit_rate, mrr, ms, compared, missed = evaluate(
                    retriever, questions, chunk_texts, k
                )
                size = f"{chunk_size} / {overlap}, {len(chunks)}"
                row = (
                    f"| {size} | {index_type} | {label} | {hit_rate:.0%} | {mrr:.2f} "
                    f"| {compared:.0%} | {ms:.0f} ms |"
                )
                print(row)
                lines.append(row)
                # Listed for the full pipeline only, to keep the report short
                if hybrid and rerank:
                    for question in missed:
                        missed_report.append(f"- {size} chunks, {index_type}: {question}")

    lines += ["", "## Questions the full pipeline (keywords + reranker) missed", ""]
    lines += missed_report or ["None."]
    RESULTS_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
