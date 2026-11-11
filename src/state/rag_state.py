"""RAG state definition for LangGraph"""

from typing import Dict, List
from pydantic import BaseModel
from langchain_core.documents import Document

class RAGState(BaseModel):
    """State object for RAG workflow"""

    question: str
    # Earlier turns, oldest first: {"role": "user" | "assistant", "content": str}
    chat_history: List[Dict[str, str]] = []
    # What is actually searched for: the question made self-contained, or a rewrite of it
    search_query: str = ""
    retrieved_docs: List[Document] = []
    docs_relevant: bool = True  # the grader's verdict on retrieved_docs
    rewrites: int = 0  # searches retried with a rewritten question
    answer: str = ""
    retrieval_time: float = 0.0  # seconds spent searching the documents
