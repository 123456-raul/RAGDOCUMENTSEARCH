"""Configuration module for Agentic RAG system"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


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

    # Document Processing
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    # Default URLs
    DEFAULT_URLS = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/"
    ]

    # Data locations
    DATA_DIR = Path(__file__).resolve().parents[2] / "data"
    URLS_FILE = DATA_DIR / "url.txt"

    @classmethod
    def get_sources(cls):
        """Return URLs from data/url.txt (or the defaults) plus the data folder if it has PDFs"""
        urls = []
        if cls.URLS_FILE.exists():
            lines = cls.URLS_FILE.read_text(encoding="utf-8").splitlines()
            urls = [line.strip() for line in lines if line.strip()]

        sources = urls or list(cls.DEFAULT_URLS)
        if any(cls.DATA_DIR.glob("*.pdf")):
            sources.append(str(cls.DATA_DIR))
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

    @classmethod
    def get_embeddings(cls):
        """Initialize and return HuggingFace embeddings (local, free)"""
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=cls.HF_EMBEDDING_MODEL)