"""Unit tests for A/B testing framework."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from qa_rag_system.ab_testing import (
    ABTestResults,
    ChunkingABTester,
    ChunkingMetrics,
    ChunkingStrategy,
    QueryTestCase,
    RetrievalMetrics,
    ResponseMetrics,
    StrategyTestResult,
)


@pytest.fixture
def mock_embeddings() -> MagicMock:
    """Create a mock embeddings model."""
    mock = MagicMock()
    mock.embed_query.return_value = [0.1] * 384  # Mock embedding vector
    return mock


@pytest.fixture
def mock_llm() -> MagicMock:
    """Create a mock LLM."""
    mock = MagicMock()
    mock.invoke.return_value.content = "This is a test answer."
    return mock


@pytest.fixture
def mock_vector_store_config() -> MagicMock:
    """Create a mock vector store config."""
    mock = MagicMock()
    mock.provider = "chroma"
    mock.index_name = None
    mock.persist_directory = None
    return mock


@pytest.fixture
def sample_documents() -> list:
    """Create sample documents for testing."""
    from langchain_core.documents import Document

    return [
        Document(
            page_content="This is a test document about Python programming.",
            metadata={"source": "test1.txt"},
        ),
        Document(
            page_content="Another document discussing machine learning concepts.",
            metadata={"source": "test2.txt"},
        ),
    ]


@pytest.fixture
def sample_test_cases() -> list[QueryTestCase]:
    """Create sample test cases."""
    return [
        QueryTestCase(
            query="What is Python?",
            expected_keywords=["Python", "programming"],
            expected_sources=["test1.txt"],
        ),
        QueryTestCase(
            query="Tell me about machine learning",
            expected_keywords=["machine", "learning"],
            expected_sources=["test2.txt"],
        ),
    ]


@pytest.fixture
def sample_strategies() -> list[ChunkingStrategy]:
    """Create sample chunking strategies."""
    return [
        ChunkingStrategy(
            name="test_recursive",
            strategy_type="recursive",
            chunk_size=500,
            chunk_overlap=100,
        ),
        ChunkingStrategy(
            name="test_token",
            strategy_type="token",
            chunk_size=500,
            chunk_overlap=100,
        ),
    ]


class TestChunkingStrategy:
    """Test ChunkingStrategy dataclass."""

    def test_create_chunker(self) -> None:
        """Test creating a chunker from strategy."""
        strategy = ChunkingStrategy(
            name="test",
            strategy_type="recursive",
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunker = strategy.create_chunker()
        assert chunker.chunk_size == 1000
        assert chunker.chunk_overlap == 200
        assert chunker.strategy == "recursive"


class TestQueryTestCase:
    """Test QueryTestCase dataclass."""

    def test_query_test_case_creation(self) -> None:
        """Test creating a query test case."""
        test_case = QueryTestCase(
            query="Test query",
            expected_keywords=["keyword1", "keyword2"],
            expected_sources=["source1.txt"],
            ground_truth_answer="Expected answer",
        )
        assert test_case.query == "Test query"
        assert test_case.expected_keywords == ["keyword1", "keyword2"]
        assert test_case.expected_sources == ["source1.txt"]
        assert test_case.ground_truth_answer == "Expected answer"


class TestChunkingABTester:
    """Test ChunkingABTester class."""

    def test_init(
        self,
        mock_embeddings: MagicMock,
        mock_llm: MagicMock,
        mock_vector_store_config: MagicMock,
    ) -> None:
        """Test AB tester initialization."""
        tester = ChunkingABTester(
            embeddings=mock_embeddings,
            llm=mock_llm,
            vector_store_config=mock_vector_store_config,
            top_k=5,
            score_threshold=0.7,
        )
        assert tester.embeddings == mock_embeddings
        assert tester.llm == mock_llm
        assert tester.top_k == 5
        assert tester.score_threshold == 0.7

    def test_calculate_chunking_metrics(
        self,
        mock_embeddings: MagicMock,
        mock_llm: MagicMock,
        mock_vector_store_config: MagicMock,
        sample_documents: list,
    ) -> None:
        """Test chunking metrics calculation."""
        tester = ChunkingABTester(
            embeddings=mock_embeddings,
            llm=mock_llm,
            vector_store_config=mock_vector_store_config,
        )

        # Create some chunks
        from langchain_core.documents import Document

        chunks = [
            Document(page_content="Short"),
            Document(page_content="Medium length chunk"),
            Document(page_content="This is a much longer chunk of text"),
        ]

        metrics = tester._calculate_chunking_metrics(chunks)

        assert metrics.total_chunks == 3
        assert metrics.avg_chunk_size > 0
        assert metrics.min_chunk_size > 0
        assert metrics.max_chunk_size > 0

    def test_calculate_semantic_similarity(
        self,
        mock_embeddings: MagicMock,
        mock_llm: MagicMock,
        mock_vector_store_config: MagicMock,
    ) -> None:
        """Test semantic similarity calculation."""
        tester = ChunkingABTester(
            embeddings=mock_embeddings,
            llm=mock_llm,
            vector_store_config=mock_vector_store_config,
        )

        # Mock embeddings to return different vectors
        def mock_embed(query: str) -> list[float]:
            if "answer" in query.lower():
                return [0.9] * 384
            return [0.1] * 384

        mock_embeddings.embed_query.side_effect = mock_embed

        similarity = tester._calculate_semantic_similarity(
            "Test answer", "Expected answer"
        )
        assert 0.0 <= similarity <= 1.0

    def test_calculate_retrieval_metrics(
        self,
        mock_embeddings: MagicMock,
        mock_llm: MagicMock,
        mock_vector_store_config: MagicMock,
        sample_test_cases: list[QueryTestCase],
    ) -> None:
        """Test retrieval metrics calculation."""
        tester = ChunkingABTester(
            embeddings=mock_embeddings,
            llm=mock_llm,
            vector_store_config=mock_vector_store_config,
        )

        from langchain_core.documents import Document

        test_case = sample_test_cases[0]
        retrieved_docs = [
            Document(
                page_content="Python is a programming language",
                metadata={"source": "test1.txt", "score": 0.8},
            ),
            Document(
                page_content="Some other content",
                metadata={"source": "other.txt", "score": 0.5},
            ),
        ]

        metrics = tester._calculate_retrieval_metrics(
            retrieved_docs, test_case.query, test_case
        )

        assert metrics.num_retrieved == 2
        assert metrics.precision_at_k >= 0.0
        assert metrics.mean_similarity_score > 0.0

    def test_save_and_load_results(
        self,
        mock_embeddings: MagicMock,
        mock_llm: MagicMock,
        mock_vector_store_config: MagicMock,
    ) -> None:
        """Test saving and loading test results."""
        tester = ChunkingABTester(
            embeddings=mock_embeddings,
            llm=mock_llm,
            vector_store_config=mock_vector_store_config,
        )

        # Create sample results
        results = ABTestResults(
            test_name="test_run",
            strategies=["strategy1", "strategy2"],
            test_cases=[
                QueryTestCase(query="Test query"),
            ],
            results={
                "strategy1": [
                    StrategyTestResult(
                        strategy_name="strategy1",
                        query="Test query",
                        retrieval_metrics=RetrievalMetrics(),
                        response_metrics=ResponseMetrics(),
                        chunking_metrics=ChunkingMetrics(),
                        answer="Test answer",
                        retrieved_docs=[],
                    )
                ],
            },
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_results.json"
            tester.save_results(results, output_path)

            assert output_path.exists()

            loaded_results = tester.load_results(output_path)
            assert loaded_results.test_name == results.test_name
            assert loaded_results.strategies == results.strategies
            assert len(loaded_results.test_cases) == len(results.test_cases)

    def test_calculate_summary_metrics(
        self,
        mock_embeddings: MagicMock,
        mock_llm: MagicMock,
        mock_vector_store_config: MagicMock,
    ) -> None:
        """Test summary metrics calculation."""
        tester = ChunkingABTester(
            embeddings=mock_embeddings,
            llm=mock_llm,
            vector_store_config=mock_vector_store_config,
        )

        results = {
            "strategy1": [
                StrategyTestResult(
                    strategy_name="strategy1",
                    query="Query 1",
                    retrieval_metrics=RetrievalMetrics(
                        precision_at_k=0.8, recall_at_k=0.6, mean_reciprocal_rank=0.7
                    ),
                    response_metrics=ResponseMetrics(
                        response_time_seconds=1.5, semantic_similarity=0.85
                    ),
                    chunking_metrics=ChunkingMetrics(total_chunks=100, avg_chunk_size=500),
                    answer="Answer 1",
                    retrieved_docs=[],
                ),
                StrategyTestResult(
                    strategy_name="strategy1",
                    query="Query 2",
                    retrieval_metrics=RetrievalMetrics(
                        precision_at_k=0.9, recall_at_k=0.7, mean_reciprocal_rank=0.8
                    ),
                    response_metrics=ResponseMetrics(
                        response_time_seconds=1.2, semantic_similarity=0.9
                    ),
                    chunking_metrics=ChunkingMetrics(total_chunks=100, avg_chunk_size=500),
                    answer="Answer 2",
                    retrieved_docs=[],
                ),
            ],
        }

        summary = tester._calculate_summary_metrics(results)

        assert "strategy1" in summary
        assert summary["strategy1"]["avg_precision_at_k"] == pytest.approx(0.85)
        assert summary["strategy1"]["avg_recall_at_k"] == pytest.approx(0.65)
        assert summary["strategy1"]["avg_mrr"] == pytest.approx(0.75)

