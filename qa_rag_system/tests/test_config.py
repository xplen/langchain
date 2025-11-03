"""Tests for configuration management."""

import os
from unittest.mock import patch

import pytest

from qa_rag_system.config import AppConfig, EmbeddingConfig, LLMConfig, VectorDBConfig


def test_llm_config_openai() -> None:
    """Test LLM config for OpenAI."""
    with patch.dict(
        os.environ,
        {
            "LLM_PROVIDER": "openai",
            "LLM_MODEL": "gpt-4",
            "OPENAI_API_KEY": "test-key",
        },
    ):
        config = LLMConfig.from_env()
        assert config.provider == "openai"
        assert config.model == "gpt-4"
        assert config.api_key == "test-key"


def test_llm_config_ollama() -> None:
    """Test LLM config for Ollama."""
    with patch.dict(
        os.environ,
        {
            "LLM_PROVIDER": "ollama",
            "LLM_MODEL": "llama3",
            "OLLAMA_BASE_URL": "http://localhost:11434",
        },
    ):
        config = LLMConfig.from_env()
        assert config.provider == "ollama"
        assert config.model == "llama3"
        assert config.base_url == "http://localhost:11434"


def test_embedding_config_openai() -> None:
    """Test embedding config for OpenAI."""
    with patch.dict(
        os.environ,
        {
            "EMBEDDING_PROVIDER": "openai",
            "EMBEDDING_MODEL": "text-embedding-3-small",
            "OPENAI_API_KEY": "test-key",
        },
    ):
        config = EmbeddingConfig.from_env()
        assert config.provider == "openai"
        assert config.model == "text-embedding-3-small"
        assert config.api_key == "test-key"


def test_embedding_config_sentence_transformers() -> None:
    """Test embedding config for Sentence Transformers."""
    with patch.dict(
        os.environ,
        {
            "EMBEDDING_PROVIDER": "sentence-transformers",
            "EMBEDDING_MODEL": "all-MiniLM-L6-v2",
        },
    ):
        config = EmbeddingConfig.from_env()
        assert config.provider == "sentence-transformers"
        assert config.model == "all-MiniLM-L6-v2"


def test_vector_db_config_chroma() -> None:
    """Test vector DB config for Chroma."""
    with patch.dict(
        os.environ,
        {
            "VECTOR_DB": "chroma",
            "CHROMA_PERSIST_DIRECTORY": "./test_db",
        },
    ):
        config = VectorDBConfig.from_env()
        assert config.provider == "chroma"
        assert config.persist_directory == "./test_db"


def test_vector_db_config_pinecone() -> None:
    """Test vector DB config for Pinecone."""
    with patch.dict(
        os.environ,
        {
            "VECTOR_DB": "pinecone",
            "PINECONE_API_KEY": "test-key",
            "PINECONE_ENVIRONMENT": "us-east-1",
            "PINECONE_INDEX_NAME": "test-index",
        },
    ):
        config = VectorDBConfig.from_env()
        assert config.provider == "pinecone"
        assert config.api_key == "test-key"
        assert config.environment == "us-east-1"
        assert config.index_name == "test-index"

