"""Tests for loading and splitting documents, source labels, and the saved index"""

import numpy as np
import pytest
from langchain_core.documents import Document

from src.document_ingestion.document_processor import (
    DocumentProcessor,
    source_label,
    source_url,
)
from src.vectorstore.index_cache import IndexCache


# ---- Source labels ----

def test_web_page_is_labelled_with_its_title():
    doc = Document(
        page_content="x",
        metadata={"source": "https://example.com/post/", "title": " A Post "},
    )
    assert source_label(doc) == "A Post"
    assert source_url(doc) == "https://example.com/post/"


def test_web_page_without_title_falls_back_to_the_site():
    doc = Document(page_content="x", metadata={"source": "https://example.com/post/"})
    assert source_label(doc) == "example.com"


def test_pdf_is_labelled_with_file_name_and_page():
    # pypdf counts pages from 0, readers count from 1
    doc = Document(page_content="x", metadata={"source": r"C:\data\paper.pdf", "page": 2})
    assert source_label(doc) == "paper.pdf, page 3"
    assert source_url(doc) == ""


def test_pdf_prefers_the_printed_page_label():
    doc = Document(
        page_content="x",
        metadata={"source": "data/paper.pdf", "page": 2, "page_label": "iv"},
    )
    assert source_label(doc) == "paper.pdf, page iv"


def test_text_file_is_labelled_with_its_name():
    doc = Document(page_content="x", metadata={"source": "uploads/notes.txt"})
    assert source_label(doc) == "notes.txt"


# ---- Loading and splitting ----

def test_text_and_markdown_files_are_loaded(tmp_path):
    (tmp_path / "a.txt").write_text("plain text", encoding="utf-8")
    (tmp_path / "b.md").write_text("# heading", encoding="utf-8")
    docs = DocumentProcessor().load_documents(
        [str(tmp_path / "a.txt"), str(tmp_path / "b.md")]
    )
    assert [d.page_content for d in docs] == ["plain text", "# heading"]


def test_unsupported_file_type_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unsupported source type"):
        DocumentProcessor().load_documents([str(tmp_path / "sheet.xlsx")])


def test_chunks_respect_the_size_limit_and_keep_metadata():
    text = " ".join(f"word{i}" for i in range(400))
    doc = Document(page_content=text, metadata={"source": "notes.txt"})
    chunks = DocumentProcessor(chunk_size=200, chunk_overlap=40).split_documents([doc])
    assert len(chunks) > 1
    assert all(len(c.page_content) <= 200 for c in chunks)
    assert all(c.metadata["source"] == "notes.txt" for c in chunks)


def test_neighbouring_chunks_overlap():
    text = " ".join(f"word{i}" for i in range(400))
    chunks = DocumentProcessor(chunk_size=200, chunk_overlap=40).split_documents(
        [Document(page_content=text)]
    )
    last_word = chunks[0].page_content.split()[-1]
    assert last_word in chunks[1].page_content.split()


# ---- Saved index ----

def test_saved_index_round_trip(tmp_path):
    cache = IndexCache(tmp_path / "cache")
    documents = [Document(page_content="hello", metadata={"source": "a.txt", "page": 1})]
    vectors = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)

    cache.save("key", documents, vectors)
    loaded_docs, loaded_vectors = cache.load("key")

    assert loaded_docs[0].page_content == "hello"
    assert loaded_docs[0].metadata == {"source": "a.txt", "page": 1}
    np.testing.assert_allclose(loaded_vectors, vectors)


def test_saved_index_is_ignored_for_another_key_or_after_clear(tmp_path):
    cache = IndexCache(tmp_path / "cache")
    cache.save("key", [Document(page_content="hello")], np.ones((1, 3), dtype=np.float32))
    assert cache.load("other key") is None
    cache.clear()
    assert cache.load("key") is None


def test_missing_saved_index_loads_as_none(tmp_path):
    assert IndexCache(tmp_path / "nothing here").load("key") is None


def test_key_changes_with_file_content_and_settings(tmp_path):
    file = tmp_path / "notes.txt"
    file.write_text("first version", encoding="utf-8")
    sources = ["https://example.com", str(file)]

    key = IndexCache.make_key(sources, chunk_size=1000)
    assert IndexCache.make_key(sources, chunk_size=1000) == key
    assert IndexCache.make_key(sources, chunk_size=500) != key

    file.write_text("second version", encoding="utf-8")
    assert IndexCache.make_key(sources, chunk_size=1000) != key
