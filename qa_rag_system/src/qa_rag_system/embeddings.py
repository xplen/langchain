"""Embedding utilities supporting multiple providers."""

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore[import-untyped]

from qa_rag_system.config import EmbeddingConfig


def create_embeddings(config: EmbeddingConfig) -> Embeddings:
    """Create an embeddings instance based on configuration.

    Args:
        config: Embedding configuration.

    Returns:
        Embeddings instance.
    """
    if config.provider == "openai":
        if not config.api_key:
            msg = "OpenAI API key is required for OpenAI embeddings"
            raise ValueError(msg)
        return OpenAIEmbeddings(
            model=config.model,
            openai_api_key=config.api_key,
        )
    elif config.provider == "sentence-transformers":
        return HuggingFaceEmbeddings(
            model_name=config.model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    else:
        msg = f"Unsupported embedding provider: {config.provider}"
        raise ValueError(msg)

