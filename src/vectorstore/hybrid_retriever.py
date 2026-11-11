"""The retriever used by the app: embedding search, keyword search, reranking.

1. Embedding search (tree or flat index) finds chunks with a similar meaning.
2. Keyword search (BM25) finds chunks containing the query's exact words.
3. The two lists are merged with reciprocal rank fusion.
4. A cross-encoder rereads the merged candidates and keeps the best few.

Steps 2 to 4 are optional; without them this is a plain embedding search.
"""

from typing import Any, Dict, List, Optional

import numpy as np
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, PrivateAttr


def reciprocal_rank_fusion(rankings: List[List[int]], c: int = 60) -> List[int]:
    """
    Merge several ranked id lists into one.

    Each list gives an id 1 / (c + rank) points, so an id ranked well by
    several lists beats one ranked top by a single list. Only ranks are used,
    which is why scores on different scales (cosine, BM25) can be combined.
    """
    points: Dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            points[doc_id] = points.get(doc_id, 0.0) + 1.0 / (c + rank)
    # sorted() is stable, so ties keep the order of the first list
    return sorted(points, key=lambda doc_id: points[doc_id], reverse=True)


class HybridRetriever(BaseRetriever):
    """LangChain retriever combining an embedding index, BM25 and a reranker"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    documents: List[Document]
    embedding: Any
    index: Any  # TreeIndex or FlatIndex
    bm25: Optional[Any] = None  # BM25Index, or None for embeddings only
    reranker: Optional[Any] = None  # Reranker, or None to skip reranking
    k: int = 4  # chunks returned
    candidate_k: int = 20  # chunks fetched before merging and reranking

    # Details of the latest search: chunks compared, candidates, scores
    _last_search: Dict[str, Any] = PrivateAttr(default_factory=dict)

    @property
    def last_search(self) -> Dict[str, Any]:
        return self._last_search

    def rank(self, query: str, k: Optional[int] = None) -> List[int]:
        """Return the ids of the best `k` chunks for the query, best first"""
        k = k or self.k
        # A wider net is only useful when a later step narrows it again
        fetch = self.candidate_k if (self.bm25 or self.reranker) else k
        fetch = max(fetch, k)

        query_vector = np.array(self.embedding.embed_query(query), dtype=np.float32)
        dense_ids, _, comparisons = self.index.search(query_vector, k=fetch)
        candidates = list(dense_ids)

        if self.bm25 is not None:
            keyword_ids, _ = self.bm25.search(query, k=fetch)
            candidates = reciprocal_rank_fusion([dense_ids, keyword_ids])[:fetch]

        scores: List[float] = []
        if self.reranker is not None:
            texts = [self.documents[i].page_content for i in candidates]
            ranked, scores = self.reranker.rerank(query, candidates, texts, top_k=k)
        else:
            ranked = candidates[:k]

        self._last_search = {
            "compared": comparisons,
            "total_chunks": len(self.documents),
            "candidates": len(candidates),
            "rerank_scores": scores,
        }
        return ranked

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return [self.documents[i] for i in self.rank(query)]
