"""Central configuration for the Dream of the Red Chamber multi-agent QA app."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)


class Config:
    """Application configuration values."""

    # Files
    BASE_DIR: Path = BASE_DIR
    NOVEL_PATH: Path = BASE_DIR / "红楼梦.txt"
    CHROMA_DIR: Path = BASE_DIR / "chroma_db"
    CHROMA_COLLECTION_NAME: str = "hongloumeng_chunks"

    # Embedding and chunking
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100
    RETRIEVAL_TOP_K: int = 5

    # DeepSeek OpenAI-compatible API
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    DEEPSEEK_MODEL_NAME: str = os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))

    # Agent behavior
    MAX_AGENT_ITERATIONS: int = 4
    MAX_HISTORY_MESSAGES: int = 8


def validate_runtime_config() -> None:
    """Validate runtime requirements before building agents."""

    if not Config.DEEPSEEK_API_KEY:
        raise RuntimeError("Missing DEEPSEEK_API_KEY in .env")
    if not Config.DEEPSEEK_BASE_URL:
        raise RuntimeError("Missing DEEPSEEK_BASE_URL in .env")
