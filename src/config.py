"""Configuration management for LLM providers and API keys."""

import os
from enum import Enum
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


class LLMProvider(Enum):
    """Supported LLM providers."""

    GEMINI = "gemini"
    OPENAI = "openai"


class LLMConfig:
    """Configuration for LLM providers."""

    def __init__(self, provider: str):
        """
        Initialize LLM configuration.

        Args:
            provider: "gemini" or "openai"

        Raises:
            ValueError: If provider is not supported or API key is missing
        """
        try:
            self.provider = LLMProvider(provider.lower())
        except ValueError:
            raise ValueError(f"Unsupported LLM provider: {provider}. Use 'gemini' or 'openai'.")

        self._load_api_key()
        self._set_model_and_params()

    def _load_api_key(self):
        """Load API key from environment."""
        if self.provider == LLMProvider.GEMINI:
            self.api_key = os.getenv("GEMINI_API_KEY")
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY not found in environment variables")
        elif self.provider == LLMProvider.OPENAI:
            self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")

    def _set_model_and_params(self):
        """Set model name and default parameters based on provider."""
        if self.provider == LLMProvider.GEMINI:
            self.base_model = "gemini/gemini-2.0-flash-lite"
            self.advanced_model = "gemini/gemini-2.0-flash"
            self.temperature = 0.2
        elif self.provider == LLMProvider.OPENAI:
            self.base_model = "gpt-5.4-mini"
            self.advanced_model = "gpt-5.4"
            self.temperature = 0.5


class QdrantConfig:
    """Configuration for Qdrant Vector Database."""

    def __init__(self):
        # Default to 'qdrant' because that's the service name in docker-compose
        self.host = os.getenv("QDRANT_HOST", "qdrant")
        self.port = int(os.getenv("QDRANT_PORT", 6333))
        self.grpc_port = int(os.getenv("QDRANT_GRPC_PORT", 6334))
        # Use gRPC by default for better performance with embeddings
        self.prefer_grpc = os.getenv("QDRANT_PREFER_GRPC", "true").lower() == "true"


def get_default_provider() -> str:
    """Get default provider from environment or fallback to gemini."""
    return os.getenv("LLM_PROVIDER", "gemini")


def load_config(provider: Optional[str] = None) -> LLMConfig:
    """
    Load LLM configuration.

    Args:
        provider: Optional provider override. If None, uses LLM_PROVIDER env var.

    Returns:
        LLMConfig instance
    """
    if provider is None:
        provider = get_default_provider()
    return LLMConfig(provider)


def get_qdrant_config() -> QdrantConfig:
    """Get Qdrant configuration instance."""
    return QdrantConfig()
