"""Configuration management for the Q&A RAG system."""

import os
from dataclasses import dataclass
from typing import Literal

from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMConfig:
    """Configuration for the Language Model."""

    provider: Literal["openai", "ollama"]
    model: str
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Load LLM configuration from environment variables."""
        provider = os.getenv("LLM_PROVIDER", "openai")
        model = os.getenv("LLM_MODEL", "gpt-4")

        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                msg = "OPENAI_API_KEY environment variable is required for OpenAI provider"
                raise ValueError(msg)
            return cls(provider=provider, model=model, api_key=api_key)
        elif provider == "ollama":
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            return cls(provider=provider, model=model, base_url=base_url)
        else:
            msg = f"Unsupported LLM provider: {provider}"
            raise ValueError(msg)


@dataclass
class EmbeddingConfig:
    """Configuration for embeddings."""

    provider: Literal["openai", "sentence-transformers"]
    model: str
    api_key: str | None = None

    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        """Load embedding configuration from environment variables."""
        provider = os.getenv("EMBEDDING_PROVIDER", "openai")
        model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                msg = "OPENAI_API_KEY environment variable is required for OpenAI embeddings"
                raise ValueError(msg)
            return cls(provider=provider, model=model, api_key=api_key)
        elif provider == "sentence-transformers":
            return cls(provider=provider, model=model)
        else:
            msg = f"Unsupported embedding provider: {provider}"
            raise ValueError(msg)


@dataclass
class VectorDBConfig:
    """Configuration for vector database."""

    provider: Literal["pinecone", "chroma"]
    index_name: str | None = None
    api_key: str | None = None
    environment: str | None = None
    persist_directory: str | None = None

    @classmethod
    def from_env(cls) -> "VectorDBConfig":
        """Load vector database configuration from environment variables."""
        provider = os.getenv("VECTOR_DB", "chroma")

        if provider == "pinecone":
            api_key = os.getenv("PINECONE_API_KEY")
            environment = os.getenv("PINECONE_ENVIRONMENT")
            index_name = os.getenv("PINECONE_INDEX_NAME", "qa-rag-index")

            if not api_key:
                msg = "PINECONE_API_KEY environment variable is required for Pinecone"
                raise ValueError(msg)
            if not environment:
                msg = "PINECONE_ENVIRONMENT environment variable is required for Pinecone"
                raise ValueError(msg)

            return cls(
                provider=provider,
                index_name=index_name,
                api_key=api_key,
                environment=environment,
            )
        elif provider == "chroma":
            persist_directory = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
            return cls(provider=provider, persist_directory=persist_directory)
        else:
            msg = f"Unsupported vector DB provider: {provider}"
            raise ValueError(msg)


@dataclass
class RAGConfig:
    """Configuration for RAG system."""

    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    score_threshold: float = 0.7
    use_hybrid_retrieval: bool = True
    semantic_weight: float = 0.5
    keyword_weight: float = 0.5

    @classmethod
    def from_env(cls) -> "RAGConfig":
        """Load RAG configuration from environment variables."""
        use_hybrid = os.getenv("USE_HYBRID_RETRIEVAL", "true").lower() == "true"
        semantic_weight = float(os.getenv("SEMANTIC_WEIGHT", "0.5"))
        keyword_weight = float(os.getenv("KEYWORD_WEIGHT", "0.5"))

        # Normalize weights
        total = semantic_weight + keyword_weight
        if total > 0:
            semantic_weight = semantic_weight / total
            keyword_weight = keyword_weight / total

        return cls(
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
            top_k=int(os.getenv("TOP_K_RETRIEVAL", "5")),
            score_threshold=float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.7")),
            use_hybrid_retrieval=use_hybrid,
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight,
        )


@dataclass
class FeedbackConfig:
    """Configuration for feedback system."""

    storage_path: str = "./feedback_data.json"
    learning_rate: float = 0.1
    min_samples: int = 5
    auto_update_weights: bool = False

    @classmethod
    def from_env(cls) -> "FeedbackConfig":
        """Load feedback configuration from environment variables."""
        return cls(
            storage_path=os.getenv("FEEDBACK_STORAGE_PATH", "./feedback_data.json"),
            learning_rate=float(os.getenv("FEEDBACK_LEARNING_RATE", "0.1")),
            min_samples=int(os.getenv("FEEDBACK_MIN_SAMPLES", "5")),
            auto_update_weights=os.getenv("AUTO_UPDATE_WEIGHTS", "false").lower() == "true",
        )


@dataclass
class AppConfig:
    """Complete application configuration."""

    llm: LLMConfig
    embedding: EmbeddingConfig
    vector_db: VectorDBConfig
    rag: RAGConfig
    feedback: FeedbackConfig

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Load complete configuration from environment variables."""
        return cls(
            llm=LLMConfig.from_env(),
            embedding=EmbeddingConfig.from_env(),
            vector_db=VectorDBConfig.from_env(),
            rag=RAGConfig.from_env(),
            feedback=FeedbackConfig.from_env(),
        )

