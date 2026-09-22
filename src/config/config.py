"""Configuration module for Agentic RAG system"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration class for RAG system"""

    # API Keys
    HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    # Model Configuration
   
    HF_LLM_REPO_ID = "Qwen/Qwen3-4B-Instruct-2507"
    HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    # Document Processing
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    # Default URLs
    DEFAULT_URLS = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/"
    ]

    @classmethod
    def get_llm(cls):
        """Initialize and return the HuggingFace-hosted LLM (free tier)"""
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