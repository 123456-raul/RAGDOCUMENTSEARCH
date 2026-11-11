"""Reranking with a cross-encoder.

The embedding search scores the question and each chunk separately, which is
fast but rough. A cross-encoder reads the question and one chunk together, so
it judges relevance much better, but it is too slow to run on every chunk. It
is therefore used only on the short list of candidates the search returns.
"""

from typing import List, Tuple


class Reranker:
    """Scores (question, chunk) pairs with a local cross-encoder model"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None  # loaded on first use, it takes a few seconds

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def score(self, query: str, texts: List[str]) -> List[float]:
        """Relevance score of each text for the query (higher is better)"""
        if not texts:
            return []
        scores = self._load().predict([(query, text) for text in texts])
        return [float(s) for s in scores]

    def rerank(
        self, query: str, ids: List[int], texts: List[str], top_k: int
    ) -> Tuple[List[int], List[float]]:
        """Return the best `top_k` ids and their scores, best first"""
        scores = self.score(query, texts)
        order = sorted(range(len(ids)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [ids[i] for i in order], [scores[i] for i in order]
