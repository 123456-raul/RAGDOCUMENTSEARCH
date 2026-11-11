"""Tree-based retriever: a hierarchical k-means index searched with beam search.

A flat (linear) retriever compares the query with every chunk. Here the chunks
are grouped into a tree of clusters. A query starts at the root, follows only
the most similar branches down to the leaves, and is compared with just the
chunks stored in those leaves.
"""

from typing import List, Optional, Tuple

import numpy as np


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale vectors to unit length so a dot product is the cosine similarity"""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


class TreeNode:
    """One node of the tree: an internal cluster or a leaf holding chunk ids"""

    def __init__(self, centroid: np.ndarray, children=None, doc_ids=None):
        self.centroid = centroid
        self.children: List["TreeNode"] = children or []
        self.doc_ids: List[int] = doc_ids or []

    @property
    def is_leaf(self) -> bool:
        return not self.children


class TreeIndex:
    """Hierarchical k-means tree over chunk embeddings"""

    def __init__(
        self,
        vectors: np.ndarray,
        branching: int = 4,
        leaf_size: int = 8,
        beam_width: int = 4,
        seed: int = 0,
    ):
        """
        Args:
            vectors: Chunk embeddings, one row per chunk
            branching: Number of children per internal node
            leaf_size: Largest number of chunks kept in one leaf
            beam_width: Branches followed per level when searching
            seed: Random seed, so the same documents give the same tree
        """
        if len(vectors) == 0:
            raise ValueError("Cannot build a tree index from zero documents.")

        self.vectors = _normalize(np.asarray(vectors, dtype=np.float32))
        self.branching = max(2, branching)
        self.leaf_size = max(1, leaf_size)
        self.beam_width = max(1, beam_width)
        self._rng = np.random.default_rng(seed)
        self.num_leaves = 0
        self.depth = 0
        self.root = self._build(np.arange(len(self.vectors)), level=0)

    def _kmeans(self, ids: np.ndarray, iterations: int = 20) -> np.ndarray:
        """Cluster the given chunks and return a cluster label for each one"""
        points = self.vectors[ids]
        k = min(self.branching, len(ids))
        centroids = points[self._rng.choice(len(ids), size=k, replace=False)]
        labels = np.zeros(len(ids), dtype=np.int64)

        for step in range(iterations):
            new_labels = np.argmax(points @ centroids.T, axis=1)
            if step > 0 and np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for c in range(k):
                members = points[labels == c]
                if len(members):
                    centroids[c] = _normalize(members.mean(axis=0))
        return labels

    def _build(self, ids: np.ndarray, level: int) -> TreeNode:
        """Recursively split chunks into clusters until they fit in a leaf"""
        centroid = _normalize(self.vectors[ids].mean(axis=0))
        self.depth = max(self.depth, level)

        if len(ids) > self.leaf_size:
            labels = self._kmeans(ids)
            groups = [ids[labels == c] for c in np.unique(labels)]
            # Identical chunks cannot be split, so they stay together in a leaf
            if len(groups) > 1:
                children = [self._build(group, level + 1) for group in groups]
                return TreeNode(centroid, children=children)

        self.num_leaves += 1
        return TreeNode(centroid, doc_ids=ids.tolist())

    def search(
        self, query_vector: np.ndarray, k: int = 4, beam_width: Optional[int] = None
    ) -> Tuple[List[int], List[float], int]:
        """
        Walk down the tree, keeping the best `beam_width` branches per level.

        Returns:
            Chunk ids, their similarity scores, and how many vectors the
            query was compared with (a flat search compares with all of them)
        """
        beam_width = beam_width or self.beam_width
        query = _normalize(np.asarray(query_vector, dtype=np.float32))
        beam: List[TreeNode] = [self.root]
        comparisons = 0

        while any(not node.is_leaf for node in beam):
            candidates: List[TreeNode] = []
            for node in beam:
                candidates.extend([node] if node.is_leaf else node.children)

            scores = np.array([float(c.centroid @ query) for c in candidates])
            comparisons += len(candidates)
            best = np.argsort(-scores)[:beam_width]
            beam = [candidates[i] for i in best]

        doc_ids = np.array([i for leaf in beam for i in leaf.doc_ids])
        scores = self.vectors[doc_ids] @ query
        comparisons += len(doc_ids)

        top = np.argsort(-scores)[:k]
        return doc_ids[top].tolist(), scores[top].tolist(), comparisons
