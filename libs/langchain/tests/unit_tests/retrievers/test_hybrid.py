"""Tests for HybridRetriever."""

import warnings

import pytest
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import InMemoryVectorStore
from typing_extensions import override

from langchain_classic.retrievers.hybrid import HybridRetriever


class MockBM25Retriever(BaseRetriever):
    """Mock BM25Retriever for testing."""

    docs: list[Document]
    k: int = 4

    @override
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        """Return top k documents based on keyword matching."""
        # Simple mock: return documents containing query words
        query_words = set(query.lower().split())
        scored_docs = []
        for doc in self.docs:
            doc_words = set(doc.page_content.lower().split())
            score = len(query_words & doc_words)
            if score > 0:
                scored_docs.append((score, doc))
        # Sort by score descending and return top k
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[: self.k]]


def test_hybrid_retriever_initialization() -> None:
    """Test that HybridRetriever initializes correctly."""
    documents = [
        Document(page_content="Python is a programming language"),
        Document(page_content="Machine learning uses algorithms"),
        Document(page_content="Python has many libraries"),
    ]

    # Create vector store
    vectorstore = InMemoryVectorStore(FakeEmbeddings(size=10))
    vectorstore.add_documents(documents)

    # Create mock BM25 retriever
    bm25_retriever = MockBM25Retriever(docs=documents)
    bm25_retriever.k = 4

    # Create hybrid retriever (should warn about non-BM25Retriever)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        hybrid_retriever = HybridRetriever(
            vectorstore=vectorstore,
            bm25_retriever=bm25_retriever,
            semantic_weight=0.6,
            keyword_weight=0.4,
        )

    assert hybrid_retriever.vectorstore == vectorstore
    assert hybrid_retriever.bm25_retriever == bm25_retriever
    assert hybrid_retriever.semantic_weight == 0.6
    assert hybrid_retriever.keyword_weight == 0.4
    assert hybrid_retriever._ensemble_retriever is not None


def test_hybrid_retriever_default_weights() -> None:
    """Test that HybridRetriever uses default weights."""
    documents = [
        Document(page_content="Python is a programming language"),
    ]

    vectorstore = InMemoryVectorStore(FakeEmbeddings(size=10))
    vectorstore.add_documents(documents)

    bm25_retriever = MockBM25Retriever(docs=documents)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        hybrid_retriever = HybridRetriever(
            vectorstore=vectorstore,
            bm25_retriever=bm25_retriever,
        )

    assert hybrid_retriever.semantic_weight == 0.5
    assert hybrid_retriever.keyword_weight == 0.5


def test_hybrid_retriever_invoke() -> None:
    """Test that HybridRetriever returns documents."""
    documents = [
        Document(page_content="Python is a programming language"),
        Document(page_content="Machine learning uses algorithms"),
        Document(page_content="Python has many libraries"),
    ]

    vectorstore = InMemoryVectorStore(FakeEmbeddings(size=10))
    vectorstore.add_documents(documents)

    bm25_retriever = MockBM25Retriever(docs=documents)
    bm25_retriever.k = 4

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        hybrid_retriever = HybridRetriever(
            vectorstore=vectorstore,
            bm25_retriever=bm25_retriever,
        )

    results = hybrid_retriever.invoke("Python")

    assert len(results) > 0
    assert all(isinstance(doc, Document) for doc in results)


def test_hybrid_retriever_with_id_key() -> None:
    """Test HybridRetriever with id_key for document deduplication."""
    documents = [
        Document(
            page_content="Python is a programming language",
            metadata={"id": "1"},
        ),
        Document(
            page_content="Machine learning uses algorithms",
            metadata={"id": "2"},
        ),
        Document(
            page_content="Python has many libraries",
            metadata={"id": "1"},
        ),
    ]

    vectorstore = InMemoryVectorStore(FakeEmbeddings(size=10))
    vectorstore.add_documents(documents)

    bm25_retriever = MockBM25Retriever(docs=documents)
    bm25_retriever.k = 4

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        hybrid_retriever = HybridRetriever(
            vectorstore=vectorstore,
            bm25_retriever=bm25_retriever,
            id_key="id",
        )

    results = hybrid_retriever.invoke("Python")

    assert len(results) > 0
    # Documents with same id should be deduplicated
    ids = [doc.metadata.get("id") for doc in results if "id" in doc.metadata]
    assert len(ids) == len(set(ids)) or len(ids) == 0


def test_hybrid_retriever_invalid_bm25_type() -> None:
    """Test that HybridRetriever raises error for invalid retriever type."""
    documents = [Document(page_content="test")]

    vectorstore = InMemoryVectorStore(FakeEmbeddings(size=10))
    vectorstore.add_documents(documents)

    # Create a non-BaseRetriever object
    invalid_retriever = "not a retriever"

    with pytest.raises(TypeError, match="bm25_retriever must be an instance of BaseRetriever"):
        HybridRetriever(
            vectorstore=vectorstore,
            bm25_retriever=invalid_retriever,
        )


def test_hybrid_retriever_custom_c_parameter() -> None:
    """Test HybridRetriever with custom RRF constant."""
    documents = [
        Document(page_content="Python is a programming language"),
    ]

    vectorstore = InMemoryVectorStore(FakeEmbeddings(size=10))
    vectorstore.add_documents(documents)

    bm25_retriever = MockBM25Retriever(docs=documents)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        hybrid_retriever = HybridRetriever(
            vectorstore=vectorstore,
            bm25_retriever=bm25_retriever,
            c=100,
        )

    assert hybrid_retriever.c == 100
    assert hybrid_retriever._ensemble_retriever is not None
    assert hybrid_retriever._ensemble_retriever.c == 100

