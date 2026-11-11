"""Flat index: FAISS compares the query with every chunk (exact search)."""

from typing import List, Tuple

import faiss
import numpy as np

from src.vectorstore.tree_retriever import _normalize


class FlatIndex:
    """Exact cosine-similarity search over all chunk embeddings"""

    def __init__(self, vectors: np.ndarray):
        if len(vectors) == 0:
            raise ValueError("Cannot build an index from zero documents.")

        self.vectors = _normalize(np.asarray(vectors, dtype=np.float32))
        self.index = faiss.IndexFlatIP(self.vectors.shape[1])
        self.index.add(self.vectors)

    def search(
        self, query_vector: np.ndarray, k: int = 4
    ) -> Tuple[List[int], List[float], int]:
        """
        Returns:
            Chunk ids, their similarity scores, and how many vectors the
            query was compared with (always all of them)
        """
        query = _normalize(np.asarray(query_vector, dtype=np.float32)).reshape(1, -1)
        k = min(k, len(self.vectors))
        scores, ids = self.index.search(query, k)
        return ids[0].tolist(), scores[0].tolist(), len(self.vectors)
