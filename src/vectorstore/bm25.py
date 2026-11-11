"""Keyword search with BM25.

Embeddings match meaning but can miss exact terms such as names, numbers and
acronyms. BM25 scores a chunk by the query words it actually contains, with
rare words counting for more than common ones.
"""

import math
import re
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lowercase the text and split it into words and numbers"""
    return _TOKEN.findall(text.lower())


class BM25Index:
    """BM25 (Okapi) index over a list of texts"""

    def __init__(self, texts: List[str], k1: float = 1.5, b: float = 0.75):
        """
        Args:
            texts: One text per chunk
            k1: How quickly repeated words stop adding to the score
            b: How strongly long chunks are penalised (0 = not at all)
        """
        self.k1 = k1
        self.b = b
        self.num_docs = len(texts)

        lengths = np.zeros(self.num_docs, dtype=np.float32)
        postings: Dict[str, Tuple[List[int], List[int]]] = {}
        for doc_id, text in enumerate(texts):
            counts = Counter(tokenize(text))
            lengths[doc_id] = sum(counts.values())
            for term, freq in counts.items():
                ids, freqs = postings.setdefault(term, ([], []))
                ids.append(doc_id)
                freqs.append(freq)

        average = float(lengths.mean()) if self.num_docs else 0.0
        # Length part of the BM25 formula, the same for every query
        self._length_norm = k1 * (1 - b + b * lengths / max(average, 1e-9))
        self._postings = {
            term: (np.array(ids, dtype=np.int64), np.array(freqs, dtype=np.float32))
            for term, (ids, freqs) in postings.items()
        }

    def _idf(self, doc_count: int) -> float:
        """Weight of a word: high when few chunks contain it"""
        return math.log(1 + (self.num_docs - doc_count + 0.5) / (doc_count + 0.5))

    def search(self, query: str, k: int = 20) -> Tuple[List[int], List[float]]:
        """Return the ids and scores of the best `k` chunks sharing a word with the query"""
        scores = np.zeros(self.num_docs, dtype=np.float32)
        for term in set(tokenize(query)):
            if term not in self._postings:
                continue
            ids, freqs = self._postings[term]
            scores[ids] += (
                self._idf(len(ids))
                * freqs
                * (self.k1 + 1)
                / (freqs + self._length_norm[ids])
            )

        # Stable sort, so equal scores keep document order
        order = np.argsort(-scores, kind="stable")[:k]
        order = order[scores[order] > 0]
        return order.tolist(), scores[order].tolist()
