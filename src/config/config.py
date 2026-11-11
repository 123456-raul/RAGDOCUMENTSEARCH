"""Configuration module for Agentic RAG system"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_flag(name: str, default: bool) -> bool:
    """Read an on/off setting such as RERANK=false from the environment"""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


class Config:
    """Configuration class for RAG system"""

    # API Keys
    HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Model Configuration
    # LLM_PROVIDER: ollama | groq | huggingface | openai
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    HF_LLM_REPO_ID = os.getenv("HF_LLM_REPO_ID", "Qwen/Qwen2.5-1.5B-Instruct")
    HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    # Document Processing (characters)
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

    # Retrieval
    # RETRIEVER_TYPE: tree (cluster tree + beam search) | flat (FAISS, compares every chunk)
    RETRIEVER_TYPE = os.getenv("RETRIEVER_TYPE", "tree").lower()
    RETRIEVER_K = 4       # chunks given to the model
    CANDIDATE_K = 20      # chunks fetched before merging and reranking
    # Add BM25 keyword search to the embedding search
    HYBRID_SEARCH = _env_flag("HYBRID_SEARCH", True)
    # Let a cross-encoder pick the best RETRIEVER_K of the candidates
    RERANK = _env_flag("RERANK", True)
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Answering
    # Ask the model whether the retrieved chunks fit the question and, if not,
    # rewrite the question and search again (one extra model call per question)
    CHECK_RELEVANCE = _env_flag("CHECK_RELEVANCE", True)
    MAX_REWRITES = 1      # searches retried with a rewritten question
    HISTORY_TURNS = 3     # earlier question/answer pairs the model sees
    TREE_BRANCHING = 4   # children per tree node
    TREE_LEAF_SIZE = 8   # most chunks stored in one leaf
    TREE_BEAM_WIDTH = 4  # branches followed per level; higher is more accurate but slower

    # Default URLs
    DEFAULT_URLS = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/"
    ]

    # Data locations
    DATA_DIR = PROJECT_ROOT / "data"
    URLS_FILE = DATA_DIR / "url.txt"

    # Files added through the app's uploader (not committed to git)
    UPLOAD_DIR = PROJECT_ROOT / "uploads"
    UPLOAD_TYPES = ("pdf", "txt", "md")

    # Saved chunks and embeddings, reused on restart (not committed to git)
    INDEX_CACHE_DIR = PROJECT_ROOT / "index_cache"

    @classmethod
    def get_uploads(cls):
        """Return the uploaded files, sorted by name"""
        if not cls.UPLOAD_DIR.is_dir():
            return []
        return sorted(
            path
            for path in cls.UPLOAD_DIR.iterdir()
            if path.is_file() and path.suffix.lower().lstrip(".") in cls.UPLOAD_TYPES
        )

    @classmethod
    def get_sources(cls):
        """Return URLs from data/url.txt (or the defaults), the data folder if it has PDFs, and uploads"""
        urls = []
        if cls.URLS_FILE.exists():
            lines = cls.URLS_FILE.read_text(encoding="utf-8").splitlines()
            urls = [line.strip() for line in lines if line.strip()]

        sources = urls or list(cls.DEFAULT_URLS)
        if any(cls.DATA_DIR.glob("*.pdf")):
            sources.append(str(cls.DATA_DIR))
        sources.extend(str(path) for path in cls.get_uploads())
        return sources

    @classmethod
    def get_llm(cls):
        """Initialize and return the LLM selected by LLM_PROVIDER"""
        if cls.LLM_PROVIDER == "ollama":
            from langchain_ollama import ChatOllama

            return ChatOllama(
                model=cls.OLLAMA_MODEL,
                temperature=0.1,
                # Keeps the model's thinking out of the answer; it also counts
                # toward num_predict, hence the higher limit
                reasoning=True,
                num_predict=2048,
            )

        if cls.LLM_PROVIDER == "groq":
            from langchain_groq import ChatGroq

            if not cls.GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY not set in .env")

            return ChatGroq(
                model=cls.GROQ_MODEL,
                temperature=0.1,
                max_tokens=512,
                api_key=cls.GROQ_API_KEY,
            )

        if cls.LLM_PROVIDER == "openai":
            from langchain_openai import ChatOpenAI

            if not cls.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not set in .env")

            return ChatOpenAI(
                model=cls.OPENAI_MODEL,
                temperature=0.1,
                max_tokens=512,
                api_key=cls.OPENAI_API_KEY,
            )

        if cls.LLM_PROVIDER != "huggingface":
            raise ValueError(f"Unknown LLM_PROVIDER: {cls.LLM_PROVIDER}")

        from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

        if not cls.HUGGINGFACEHUB_API_TOKEN:
            raise ValueError("HUGGINGFACEHUB_API_TOKEN not set in .env")

        endpoint = HuggingFaceEndpoint(
            repo_id=cls.HF_LLM_REPO_ID,
            task="text-generation",
            max_new_tokens=512,
            temperature=0.1,
            huggingfacehub_api_token=cls.HUGGINGFACEHUB_API_TOKEN,
        )
        return ChatHuggingFace(llm=endpoint)
