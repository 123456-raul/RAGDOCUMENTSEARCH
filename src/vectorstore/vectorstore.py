"""Vector store module for document embedding and retrieval"""

from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.config.config import Config
from src.vectorstore.bm25 import BM25Index
from src.vectorstore.flat_index import FlatIndex
from src.vectorstore.hybrid_retriever import HybridRetriever
from src.vectorstore.index_cache import IndexCache
from src.vectorstore.reranker import Reranker
from src.vectorstore.tree_retriever import TreeIndex


@lru_cache(maxsize=None)
def get_embedding_model(model_name: str) -> HuggingFaceEmbeddings:
    """Load the embedding model once and share it, since loading is slow"""
    return HuggingFaceEmbeddings(model_name=model_name)


@lru_cache(maxsize=None)
def get_reranker(model_name: str) -> Reranker:
    """Share one reranker, so its model is loaded only once"""
    return Reranker(model_name)


class VectorStore:
    """Manages vector store operations"""

    def __init__(self):
        """Initialize vector store with HuggingFace embeddings"""
        self.embedding = get_embedding_model(Config.HF_EMBEDDING_MODEL)
        self.cache = IndexCache(Config.INDEX_CACHE_DIR)
        self.retriever = None

    def load_or_build(self, sources: List[str], doc_processor) -> Tuple[int, bool]:
        """
        Create the retriever, reusing the index saved on disk when possible

        The saved index is used only if the sources, the local files and the
        chunking settings are the same as when it was saved.

        Args:
            sources: URLs, PDF folders or file paths to index
            doc_processor: DocumentProcessor used to load and split the sources

        Returns:
            Number of chunks, and whether they came from the saved index
        """
        key = IndexCache.make_key(
            sources,
            chunk_size=doc_processor.chunk_size,
            chunk_overlap=doc_processor.chunk_overlap,
            embedding_model=Config.HF_EMBEDDING_MODEL,
        )

        saved = self.cache.load(key)
        if saved is not None:
            documents, vectors = saved
            self.create_vectorstore(documents, vectors)
            return len(documents), True

        documents = doc_processor.process_urls(sources)
        vectors = self.embed(documents)
        self.create_vectorstore(documents, vectors)
        self.cache.save(key, documents, vectors)
        return len(documents), False

    def embed(self, documents: List[Document]) -> np.ndarray:
        """Embed the documents, one row per document"""
        texts = [doc.page_content for doc in documents]
        return np.array(self.embedding.embed_documents(texts), dtype=np.float32)

    def build_retriever(
        self,
        documents: List[Document],
        vectors: np.ndarray,
        retriever_type: Optional[str] = None,
        hybrid: Optional[bool] = None,
        rerank: Optional[bool] = None,
    ) -> HybridRetriever:
        """
        Build a retriever over the documents

        Settings left as None come from Config. `retriever_type` picks the
        embedding search: "tree" walks a cluster tree, "flat" compares with
        every chunk (FAISS). `hybrid` adds BM25 keyword search, and `rerank`
        adds the cross-encoder.
        """
        retriever_type = retriever_type or Config.RETRIEVER_TYPE
        hybrid = Config.HYBRID_SEARCH if hybrid is None else hybrid
        rerank = Config.RERANK if rerank is None else rerank

        if retriever_type == "tree":
            index = TreeIndex(
                vectors,
                branching=Config.TREE_BRANCHING,
                leaf_size=Config.TREE_LEAF_SIZE,
                beam_width=Config.TREE_BEAM_WIDTH,
            )
        elif retriever_type == "flat":
            index = FlatIndex(vectors)
        else:
            raise ValueError(f"Unknown RETRIEVER_TYPE: {retriever_type}")

        bm25 = BM25Index([doc.page_content for doc in documents]) if hybrid else None
        reranker = get_reranker(Config.RERANKER_MODEL) if rerank else None

        return HybridRetriever(
            documents=documents,
            embedding=self.embedding,
            index=index,
            bm25=bm25,
            reranker=reranker,
            k=Config.RETRIEVER_K,
            candidate_k=Config.CANDIDATE_K,
        )

    def create_vectorstore(
        self, documents: List[Document], vectors: Optional[np.ndarray] = None
    ):
        """
        Create the app's retriever from documents

        Args:
            documents: List of documents to index
            vectors: Their embeddings, if already computed
        """
        if not documents:
            raise ValueError("No document chunks to index. Check the sources.")

        if vectors is None:
            vectors = self.embed(documents)

        self.retriever = self.build_retriever(documents, vectors)

    def warm_up(self) -> None:
        """Run one search so the models are loaded before the first real question"""
        self.get_retriever().rank("warm up")

    def get_retriever(self):
        """
        Get the retriever instance

        Returns:
            Retriever instance
        """
        if self.retriever is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        return self.retriever

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """
        Retrieve relevant documents for a query

        Args:
            query: Search query
            k: Number of documents to retrieve

        Returns:
            List of relevant documents
        """
        retriever = self.get_retriever()
        return [retriever.documents[i] for i in retriever.rank(query, k=k)]
