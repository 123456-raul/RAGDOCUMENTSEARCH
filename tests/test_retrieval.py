"""Tests for the search building blocks: BM25, the indexes, fusion and the retriever"""

import numpy as np
import pytest
from langchain_core.documents import Document

from src.vectorstore.bm25 import BM25Index, tokenize
from src.vectorstore.flat_index import FlatIndex
from src.vectorstore.hybrid_retriever import HybridRetriever, reciprocal_rank_fusion
from src.vectorstore.tree_retriever import TreeIndex

TEXTS = [
    "The Transformer uses multi-head attention with eight heads.",
    "FAISS is a library for similarity search over dense vectors.",
    "The model was trained on 8 NVIDIA P100 GPUs for 12 hours.",
    "Bananas are yellow and grow in warm climates.",
]


class FakeEmbedding:
    """Gives each query a fixed vector, so tests need no model"""

    def __init__(self, vectors_by_query):
        self.vectors_by_query = vectors_by_query

    def embed_query(self, query):
        return self.vectors_by_query[query]


class FakeReranker:
    """Prefers texts containing a given word"""

    def __init__(self, word):
        self.word = word

    def rerank(self, query, ids, texts, top_k):
        scores = [1.0 if self.word in text else 0.0 for text in texts]
        order = sorted(range(len(ids)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [ids[i] for i in order], [scores[i] for i in order]


def random_vectors(count=200, dim=16, seed=1):
    return np.random.default_rng(seed).normal(size=(count, dim)).astype(np.float32)


# ---- BM25 ----

def test_tokenize_lowercases_and_keeps_numbers():
    assert tokenize("8 NVIDIA P100 GPUs!") == ["8", "nvidia", "p100", "gpus"]


def test_bm25_finds_exact_term():
    ids, scores = BM25Index(TEXTS).search("P100", k=2)
    assert ids == [2]
    assert scores[0] > 0


def test_bm25_ranks_rare_words_above_common_ones():
    # "the" is in three texts, "faiss" in one, so the FAISS text must win
    ids, _ = BM25Index(TEXTS).search("the faiss", k=4)
    assert ids[0] == 1


def test_bm25_returns_nothing_without_shared_words():
    assert BM25Index(TEXTS).search("quantum chromodynamics") == ([], [])


# ---- Indexes ----

def test_flat_index_returns_the_vector_itself_first():
    vectors = random_vectors()
    ids, scores, compared = FlatIndex(vectors).search(vectors[17], k=3)
    assert ids[0] == 17
    assert scores[0] == pytest.approx(1.0, abs=1e-4)
    assert compared == len(vectors)


def test_tree_index_keeps_every_chunk_in_exactly_one_leaf():
    vectors = random_vectors()
    tree = TreeIndex(vectors, branching=4, leaf_size=8)

    def leaf_ids(node):
        if node.is_leaf:
            return list(node.doc_ids)
        return [i for child in node.children for i in leaf_ids(child)]

    assert sorted(leaf_ids(tree.root)) == list(range(len(vectors)))


def test_tree_search_compares_fewer_vectors_than_flat():
    vectors = random_vectors()
    tree = TreeIndex(vectors, branching=4, leaf_size=8, beam_width=2)
    ids, _, compared = tree.search(vectors[17], k=3)
    assert ids[0] == 17
    assert compared < len(vectors)


def test_tree_with_a_wide_beam_matches_flat_search():
    vectors = random_vectors()
    query = random_vectors(count=1, seed=2)[0]
    flat_ids, _, _ = FlatIndex(vectors).search(query, k=5)
    tree_ids, _, _ = TreeIndex(vectors, beam_width=10_000).search(query, k=5)
    assert tree_ids == flat_ids


def test_indexes_reject_empty_input():
    empty = np.zeros((0, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        TreeIndex(empty)
    with pytest.raises(ValueError):
        FlatIndex(empty)


# ---- Fusion and the retriever ----

def test_fusion_prefers_ids_ranked_well_by_both_lists():
    # 2 is second in both lists, which beats being first in only one
    assert reciprocal_rank_fusion([[1, 2, 3], [4, 2, 5]])[0] == 2


def test_fusion_keeps_ids_found_by_only_one_list():
    assert set(reciprocal_rank_fusion([[1, 2], [3]])) == {1, 2, 3}


def make_retriever(**kwargs):
    documents = [Document(page_content=text) for text in TEXTS]
    # One axis per text; the query "fruit" points at the banana text
    vectors = np.eye(len(TEXTS), dtype=np.float32)
    embedding = FakeEmbedding({
        "fruit": [0, 0, 0, 1],
        "P100 fruit": [0, 0, 0, 1],
    })
    return HybridRetriever(
        documents=documents, embedding=embedding, index=FlatIndex(vectors), **kwargs
    )


def test_retriever_with_embeddings_only_follows_the_index():
    retriever = make_retriever(k=1)
    assert retriever.rank("fruit") == [3]
    assert retriever.invoke("fruit")[0].page_content == TEXTS[3]


def test_keyword_search_adds_chunks_the_embedding_search_ranks_low():
    # The embedding puts the GPU text last; the keyword "P100" pulls it up
    without = make_retriever(k=2).rank("P100 fruit")
    with_keywords = make_retriever(k=2, bm25=BM25Index(TEXTS)).rank("P100 fruit")
    assert 2 not in without
    assert sorted(with_keywords) == [2, 3]


def test_reranker_has_the_final_say():
    retriever = make_retriever(k=1, candidate_k=4, reranker=FakeReranker("FAISS"))
    assert retriever.rank("fruit") == [1]
    assert retriever.last_search["candidates"] == 4
