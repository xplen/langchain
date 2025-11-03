"""Vector store abstraction supporting multiple providers."""

from langchain_chroma import Chroma
from langchain_community.vectorstores import Pinecone  # type: ignore[import-untyped]
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from qa_rag_system.config import VectorDBConfig


def create_vector_store(
    config: VectorDBConfig,
    embeddings: Embeddings,
    documents: list[Document] | None = None,
) -> VectorStore:
    """Create a vector store instance based on configuration.

    Args:
        config: Vector database configuration.
        embeddings: Embeddings instance to use.
        documents: Optional documents to initialize the store with.

    Returns:
        VectorStore instance.
    """
    if config.provider == "pinecone":
        if not config.api_key or not config.environment or not config.index_name:
            msg = "Pinecone requires API key, environment, and index name"
            raise ValueError(msg)

        if documents:
            return Pinecone.from_documents(
                documents=documents,
                embedding=embeddings,
                index_name=config.index_name,
            )
        else:
            return Pinecone.from_existing_index(
                index_name=config.index_name,
                embedding=embeddings,
            )

    elif config.provider == "chroma":
        collection_name = config.index_name or "qa_rag_collection"

        if documents:
            return Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                collection_name=collection_name,
                persist_directory=config.persist_directory,
            )
        else:
            return Chroma(
                embedding_function=embeddings,
                collection_name=collection_name,
                persist_directory=config.persist_directory,
            )
    else:
        msg = f"Unsupported vector DB provider: {config.provider}"
        raise ValueError(msg)

