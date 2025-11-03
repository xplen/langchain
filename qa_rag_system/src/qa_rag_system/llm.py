"""LLM utilities supporting multiple providers."""

from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from qa_rag_system.config import LLMConfig


def create_llm(config: LLMConfig) -> BaseChatModel:
    """Create an LLM instance based on configuration.

    Args:
        config: LLM configuration.

    Returns:
        BaseChatModel instance.
    """
    if config.provider == "openai":
        if not config.api_key:
            msg = "OpenAI API key is required for OpenAI provider"
            raise ValueError(msg)
        return ChatOpenAI(
            model=config.model,
            api_key=config.api_key,
            temperature=config.temperature,
        )
    elif config.provider == "ollama":
        base_url = config.base_url or "http://localhost:11434"
        return ChatOllama(
            model=config.model,
            base_url=base_url,
            temperature=config.temperature,
        )
    else:
        msg = f"Unsupported LLM provider: {config.provider}"
        raise ValueError(msg)

