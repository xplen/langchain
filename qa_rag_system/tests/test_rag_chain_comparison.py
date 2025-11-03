"""Tests for RAG chain comparison functionality."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever

from qa_rag_system.rag_chain import RAGChain


@pytest.fixture
def mock_llm() -> BaseChatModel:
    """Create a mock LLM for testing."""
    llm = MagicMock(spec=BaseChatModel)
    llm.invoke = MagicMock(return_value=MagicMock(content="Mock comparison answer"))
    return llm


@pytest.fixture
def mock_retriever() -> BaseRetriever:
    """Create a mock retriever for testing."""
    retriever = MagicMock(spec=BaseRetriever)
    retriever.invoke = MagicMock(
        return_value=[
            Document(
                page_content="Test document about AWS Lambda",
                metadata={"source": "lambda_doc.pdf", "chunk_index": 0},
            ),
            Document(
                page_content="Test document about Google Cloud Functions",
                metadata={"source": "gcf_doc.pdf", "chunk_index": 0},
            ),
        ]
    )
    retriever.ainvoke = AsyncMock(
        return_value=[
            Document(
                page_content="Test document about AWS Lambda",
                metadata={"source": "lambda_doc.pdf", "chunk_index": 0},
            ),
            Document(
                page_content="Test document about Google Cloud Functions",
                metadata={"source": "gcf_doc.pdf", "chunk_index": 0},
            ),
        ]
    )
    return retriever


@pytest.fixture
def rag_chain(mock_llm: BaseChatModel, mock_retriever: BaseRetriever) -> RAGChain:
    """Create a RAGChain instance for testing."""
    return RAGChain(
        llm=mock_llm,
        retriever=mock_retriever,
        top_k=5,
        score_threshold=0.7,
    )


class TestExtractComparisonEntities:
    """Test entity extraction from comparison queries."""

    def test_extract_entities_vs(self, rag_chain: RAGChain) -> None:
        """Test extraction with 'vs' pattern."""
        query = "Compare AWS Lambda vs Google Cloud Functions"
        entities = rag_chain._extract_comparison_entities(query)
        assert entities == ["AWS Lambda", "Google Cloud Functions"]

    def test_extract_entities_versus(self, rag_chain: RAGChain) -> None:
        """Test extraction with 'versus' pattern."""
        query = "Compare AWS Lambda versus Google Cloud Functions"
        entities = rag_chain._extract_comparison_entities(query)
        assert entities == ["AWS Lambda", "Google Cloud Functions"]

    def test_extract_entities_and(self, rag_chain: RAGChain) -> None:
        """Test extraction with 'and' pattern."""
        query = "Compare AWS Lambda and Google Cloud Functions"
        entities = rag_chain._extract_comparison_entities(query)
        assert entities == ["AWS Lambda", "Google Cloud Functions"]

    def test_extract_entities_difference_between(self, rag_chain: RAGChain) -> None:
        """Test extraction with 'difference between' pattern."""
        query = "What is the difference between AWS Lambda and Google Cloud Functions"
        entities = rag_chain._extract_comparison_entities(query)
        assert entities == ["AWS Lambda", "Google Cloud Functions"]

    def test_extract_entities_case_insensitive(self, rag_chain: RAGChain) -> None:
        """Test extraction is case insensitive."""
        query = "COMPARE AWS LAMBDA VS GOOGLE CLOUD FUNCTIONS"
        entities = rag_chain._extract_comparison_entities(query)
        assert entities == ["AWS LAMBDA", "GOOGLE CLOUD FUNCTIONS"]

    def test_extract_entities_with_punctuation(self, rag_chain: RAGChain) -> None:
        """Test extraction handles trailing punctuation."""
        query = "Compare AWS Lambda vs Google Cloud Functions."
        entities = rag_chain._extract_comparison_entities(query)
        assert entities == ["AWS Lambda", "Google Cloud Functions"]

    def test_extract_entities_no_match(self, rag_chain: RAGChain) -> None:
        """Test extraction returns None for non-comparison queries."""
        query = "What is AWS Lambda?"
        entities = rag_chain._extract_comparison_entities(query)
        assert entities is None


class TestFormatComparisonDocuments:
    """Test document formatting for comparison."""

    def test_format_comparison_documents(
        self, rag_chain: RAGChain, mock_retriever: BaseRetriever
    ) -> None:
        """Test formatting documents grouped by entity."""
        entity_docs = {
            "AWS Lambda": [
                Document(
                    page_content="Lambda is a serverless compute service",
                    metadata={"source": "lambda_doc.pdf", "chunk_index": 0},
                )
            ],
            "Google Cloud Functions": [
                Document(
                    page_content="Cloud Functions is a serverless execution environment",
                    metadata={"source": "gcf_doc.pdf", "chunk_index": 0},
                )
            ],
        }

        formatted = rag_chain._format_comparison_documents(entity_docs)

        assert "=== DOCUMENTS ABOUT: AWS LAMBDA ===" in formatted
        assert "=== DOCUMENTS ABOUT: GOOGLE CLOUD FUNCTIONS ===" in formatted
        assert "Lambda is a serverless compute service" in formatted
        assert "Cloud Functions is a serverless execution environment" in formatted

    def test_format_comparison_documents_empty(
        self, rag_chain: RAGChain
    ) -> None:
        """Test formatting with empty document dictionary."""
        entity_docs: dict[str, list[Document]] = {}
        formatted = rag_chain._format_comparison_documents(entity_docs)
        assert formatted == ""

    def test_format_comparison_documents_no_docs(
        self, rag_chain: RAGChain
    ) -> None:
        """Test formatting with entity having no documents."""
        entity_docs = {
            "AWS Lambda": [],
            "Google Cloud Functions": [
                Document(
                    page_content="Cloud Functions info",
                    metadata={"source": "gcf_doc.pdf", "chunk_index": 0},
                )
            ],
        }

        formatted = rag_chain._format_comparison_documents(entity_docs)

        assert "=== DOCUMENTS ABOUT: AWS LAMBDA ===" not in formatted
        assert "=== DOCUMENTS ABOUT: GOOGLE CLOUD FUNCTIONS ===" in formatted


class TestCompare:
    """Test comparison functionality."""

    def test_compare_with_explicit_entities(
        self, rag_chain: RAGChain, mock_llm: BaseChatModel, mock_retriever: BaseRetriever
    ) -> None:
        """Test comparison with explicitly provided entities."""
        query = "Compare these services"
        entities = ["AWS Lambda", "Google Cloud Functions"]

        result = rag_chain.compare(query, entities)

        assert "answer" in result
        assert result["entities"] == entities
        assert "sources" in result
        assert "num_sources" in result
        assert "sources_per_entity" in result
        assert len(result["sources"]) > 0
        mock_retriever.invoke.assert_called()

    def test_compare_with_extracted_entities(
        self, rag_chain: RAGChain, mock_llm: BaseChatModel, mock_retriever: BaseRetriever
    ) -> None:
        """Test comparison with entities extracted from query."""
        query = "Compare AWS Lambda vs Google Cloud Functions"

        result = rag_chain.compare(query)

        assert "answer" in result
        assert result["entities"] == ["AWS Lambda", "Google Cloud Functions"]
        assert "sources" in result
        assert len(result["sources"]) > 0

    def test_compare_invalid_query_no_entities(
        self, rag_chain: RAGChain
    ) -> None:
        """Test comparison fails when entities cannot be extracted."""
        query = "What is AWS Lambda?"

        with pytest.raises(ValueError, match="Could not extract entities"):
            rag_chain.compare(query)

    def test_compare_insufficient_entities(self, rag_chain: RAGChain) -> None:
        """Test comparison fails with less than 2 entities."""
        query = "Compare services"
        entities = ["AWS Lambda"]

        with pytest.raises(ValueError, match="at least 2 entities"):
            rag_chain.compare(query, entities)

    def test_compare_retrieves_documents_per_entity(
        self, rag_chain: RAGChain, mock_retriever: BaseRetriever
    ) -> None:
        """Test that documents are retrieved separately for each entity."""
        query = "Compare AWS Lambda vs Google Cloud Functions"
        entities = ["AWS Lambda", "Google Cloud Functions"]

        rag_chain.compare(query, entities)

        # Should be called once per entity
        assert mock_retriever.invoke.call_count == len(entities)

    def test_compare_sources_grouped_by_entity(
        self, rag_chain: RAGChain, mock_retriever: BaseRetriever
    ) -> None:
        """Test that sources are grouped by entity."""
        query = "Compare AWS Lambda vs Google Cloud Functions"
        entities = ["AWS Lambda", "Google Cloud Functions"]

        result = rag_chain.compare(query, entities)

        assert all("entity" in source for source in result["sources"])
        entity_names = {source["entity"] for source in result["sources"]}
        assert entity_names == set(entities)


class TestACompare:
    """Test async comparison functionality."""

    @pytest.mark.asyncio
    async def test_acompare_with_explicit_entities(
        self, rag_chain: RAGChain, mock_llm: BaseChatModel, mock_retriever: BaseRetriever
    ) -> None:
        """Test async comparison with explicitly provided entities."""
        query = "Compare these services"
        entities = ["AWS Lambda", "Google Cloud Functions"]

        result = await rag_chain.acompare(query, entities)

        assert "answer" in result
        assert result["entities"] == entities
        assert "sources" in result
        assert "num_sources" in result
        assert "sources_per_entity" in result
        mock_retriever.ainvoke.assert_called()

    @pytest.mark.asyncio
    async def test_acompare_with_extracted_entities(
        self, rag_chain: RAGChain, mock_llm: BaseChatModel, mock_retriever: BaseRetriever
    ) -> None:
        """Test async comparison with entities extracted from query."""
        query = "Compare AWS Lambda vs Google Cloud Functions"

        result = await rag_chain.acompare(query)

        assert "answer" in result
        assert result["entities"] == ["AWS Lambda", "Google Cloud Functions"]
        assert "sources" in result

    @pytest.mark.asyncio
    async def test_acompare_invalid_query_no_entities(
        self, rag_chain: RAGChain
    ) -> None:
        """Test async comparison fails when entities cannot be extracted."""
        query = "What is AWS Lambda?"

        with pytest.raises(ValueError, match="Could not extract entities"):
            await rag_chain.acompare(query)

    @pytest.mark.asyncio
    async def test_acompare_retrieves_documents_per_entity(
        self, rag_chain: RAGChain, mock_retriever: BaseRetriever
    ) -> None:
        """Test that documents are retrieved separately for each entity in async mode."""
        query = "Compare AWS Lambda vs Google Cloud Functions"
        entities = ["AWS Lambda", "Google Cloud Functions"]

        await rag_chain.acompare(query, entities)

        # Should be called once per entity
        assert mock_retriever.ainvoke.call_count == len(entities)

