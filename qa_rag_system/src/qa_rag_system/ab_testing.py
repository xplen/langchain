"""A/B testing framework for comparing chunking strategies with measurable metrics."""

import json
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from qa_rag_system.document_loader import DocumentChunker
from qa_rag_system.rag_chain import RAGChain


@dataclass
class ChunkingStrategy:
    """Configuration for a chunking strategy to test."""

    name: str
    strategy_type: str  # 'recursive', 'token', 'character', etc.
    chunk_size: int
    chunk_overlap: int
    kwargs: dict[str, Any] = field(default_factory=dict)

    def create_chunker(self) -> DocumentChunker:
        """Create a DocumentChunker instance for this strategy.

        Returns:
            DocumentChunker configured with this strategy's parameters.
        """
        return DocumentChunker(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            strategy=self.strategy_type,
        )


@dataclass
class QueryTestCase:
    """A test case with query and expected information."""

    query: str
    expected_keywords: list[str] | None = None
    expected_sources: list[str] | None = None
    ground_truth_answer: str | None = None


@dataclass
class RetrievalMetrics:
    """Metrics for document retrieval performance."""

    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    mean_reciprocal_rank: float = 0.0
    mean_similarity_score: float = 0.0
    num_retrieved: int = 0
    similarity_scores: list[float] = field(default_factory=list)


@dataclass
class ResponseMetrics:
    """Metrics for response quality."""

    answer_length: int = 0
    num_sources_cited: int = 0
    semantic_similarity: float = 0.0
    response_time_seconds: float = 0.0


@dataclass
class ChunkingMetrics:
    """Metrics related to chunking statistics."""

    total_chunks: int = 0
    avg_chunk_size: float = 0.0
    median_chunk_size: float = 0.0
    min_chunk_size: int = 0
    max_chunk_size: int = 0
    chunk_size_std: float = 0.0


@dataclass
class StrategyTestResult:
    """Results for a single strategy test."""

    strategy_name: str
    query: str
    retrieval_metrics: RetrievalMetrics
    response_metrics: ResponseMetrics
    chunking_metrics: ChunkingMetrics
    answer: str
    retrieved_docs: list[dict[str, Any]]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ABTestResults:
    """Complete A/B test results for comparison."""

    test_name: str
    strategies: list[str]
    test_cases: list[QueryTestCase]
    results: dict[str, list[StrategyTestResult]]  # strategy_name -> list of results
    summary_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class ChunkingABTester:
    """A/B testing framework for comparing chunking strategies."""

    def __init__(
        self,
        embeddings: Embeddings,
        llm: BaseChatModel,
        vector_store_config: Any,
        top_k: int = 5,
        score_threshold: float = 0.7,
    ) -> None:
        """Initialize the A/B tester.

        Args:
            embeddings: Embeddings model to use for vectorization.
            llm: Language model for generating responses.
            vector_store_config: Configuration for vector store creation.
            top_k: Number of documents to retrieve.
            score_threshold: Minimum similarity score threshold.
        """
        self.embeddings = embeddings
        self.llm = llm
        self.vector_store_config = vector_store_config
        self.top_k = top_k
        self.score_threshold = score_threshold

    def _calculate_retrieval_metrics(
        self,
        retrieved_docs: list[Document],
        query: str,
        test_case: QueryTestCase,
    ) -> RetrievalMetrics:
        """Calculate retrieval performance metrics.

        Args:
            retrieved_docs: List of retrieved documents.
            query: The query string.
            test_case: Test case with expected information.

        Returns:
            RetrievalMetrics with calculated values.
        """
        metrics = RetrievalMetrics()

        if not retrieved_docs:
            return metrics

        metrics.num_retrieved = len(retrieved_docs)

        # Extract similarity scores if available
        similarity_scores: list[float] = []
        for doc in retrieved_docs:
            if "score" in doc.metadata:
                score = doc.metadata["score"]
                # Convert distance to similarity if needed
                similarity_scores.append(1.0 - score if score < 1.0 else score)

        if similarity_scores:
            metrics.similarity_scores = similarity_scores
            metrics.mean_similarity_score = float(np.mean(similarity_scores))

        # Calculate precision/recall if expected keywords or sources provided
        if test_case.expected_keywords:
            # Simple keyword-based precision
            relevant_count = 0
            for doc in retrieved_docs:
                content_lower = doc.page_content.lower()
                if any(
                    keyword.lower() in content_lower
                    for keyword in test_case.expected_keywords
                ):
                    relevant_count += 1

            metrics.precision_at_k = relevant_count / len(retrieved_docs) if retrieved_docs else 0.0

        if test_case.expected_sources:
            # Calculate recall based on expected sources
            retrieved_sources = {
                doc.metadata.get("source", "") for doc in retrieved_docs
            }
            expected_sources = set(test_case.expected_sources)
            matched_sources = retrieved_sources.intersection(expected_sources)

            metrics.precision_at_k = (
                len(matched_sources) / len(retrieved_docs) if retrieved_docs else 0.0
            )
            metrics.recall_at_k = (
                len(matched_sources) / len(expected_sources)
                if expected_sources
                else 0.0
            )

            # Calculate MRR
            for rank, doc in enumerate(retrieved_docs, 1):
                source = doc.metadata.get("source", "")
                if source in expected_sources:
                    metrics.mean_reciprocal_rank = 1.0 / rank
                    break

        return metrics

    def _calculate_chunking_metrics(
        self, chunks: list[Document]
    ) -> ChunkingMetrics:
        """Calculate chunking statistics.

        Args:
            chunks: List of chunked documents.

        Returns:
            ChunkingMetrics with calculated statistics.
        """
        if not chunks:
            return ChunkingMetrics()

        chunk_sizes = [len(chunk.page_content) for chunk in chunks]

        return ChunkingMetrics(
            total_chunks=len(chunks),
            avg_chunk_size=float(np.mean(chunk_sizes)),
            median_chunk_size=float(np.median(chunk_sizes)),
            min_chunk_size=min(chunk_sizes),
            max_chunk_size=max(chunk_sizes),
            chunk_size_std=float(np.std(chunk_sizes)),
        )

    def _calculate_semantic_similarity(
        self, answer: str, ground_truth: str | None
    ) -> float:
        """Calculate semantic similarity between answer and ground truth.

        Args:
            answer: Generated answer.
            ground_truth: Expected answer (optional).

        Returns:
            Similarity score between 0 and 1.
        """
        if not ground_truth:
            return 0.0

        try:
            # Embed both texts
            answer_embedding = self.embeddings.embed_query(answer)
            truth_embedding = self.embeddings.embed_query(ground_truth)

            # Calculate cosine similarity
            answer_vec = np.array(answer_embedding)
            truth_vec = np.array(truth_embedding)

            dot_product = np.dot(answer_vec, truth_vec)
            norm_a = np.linalg.norm(answer_vec)
            norm_t = np.linalg.norm(truth_vec)

            if norm_a == 0 or norm_t == 0:
                return 0.0

            similarity = dot_product / (norm_a * norm_t)
            return float(max(0.0, min(1.0, similarity)))
        except Exception:
            return 0.0

    def test_strategy(
        self,
        strategy: ChunkingStrategy,
        documents: list[Document],
        test_cases: list[QueryTestCase],
    ) -> list[StrategyTestResult]:
        """Test a single chunking strategy against test cases.

        Args:
            strategy: Chunking strategy to test.
            documents: Original documents to chunk.
            test_cases: List of query test cases.

        Returns:
            List of test results for each query.
        """
        from qa_rag_system.vector_store import create_vector_store

        # Chunk documents with this strategy
        chunker = strategy.create_chunker()
        chunks = chunker.chunk_documents(documents)

        # Calculate chunking metrics
        chunking_metrics = self._calculate_chunking_metrics(chunks)

        # Create vector store with chunks
        # Use unique collection/index name for each strategy to avoid conflicts
        original_index_name = self.vector_store_config.index_name
        unique_index_name = (
            f"{original_index_name}_{strategy.name}"
            if original_index_name
            else f"ab_test_{strategy.name}"
        )

        # Create a modified config with unique index name
        test_config = replace(
            self.vector_store_config, index_name=unique_index_name
        )

        vector_store = create_vector_store(
            config=test_config,
            embeddings=self.embeddings,
            documents=chunks,
        )

        # Create retriever
        retriever = vector_store.as_retriever(
            search_kwargs={"k": self.top_k}
        )

        # Create RAG chain
        rag_chain = RAGChain(
            llm=self.llm,
            retriever=retriever,
            top_k=self.top_k,
            score_threshold=self.score_threshold,
        )

        results: list[StrategyTestResult] = []

        # Test each query
        for test_case in test_cases:
            start_time = time.time()

            # Get response
            response = rag_chain.invoke(test_case.query)

            response_time = time.time() - start_time

            # Retrieve documents again for metrics
            retrieved_docs = retriever.invoke(test_case.query)

            # Calculate metrics
            retrieval_metrics = self._calculate_retrieval_metrics(
                retrieved_docs, test_case.query, test_case
            )

            semantic_similarity = self._calculate_semantic_similarity(
                response["answer"], test_case.ground_truth_answer
            )

            response_metrics = ResponseMetrics(
                answer_length=len(response["answer"]),
                num_sources_cited=response["num_sources"],
                semantic_similarity=semantic_similarity,
                response_time_seconds=response_time,
            )

            # Format retrieved docs for storage
            retrieved_docs_dict = [
                {
                    "source": doc.metadata.get("source", "Unknown"),
                    "chunk_index": doc.metadata.get("chunk_index"),
                    "content_preview": doc.page_content[:200],
                    "score": doc.metadata.get("score"),
                }
                for doc in retrieved_docs
            ]

            result = StrategyTestResult(
                strategy_name=strategy.name,
                query=test_case.query,
                retrieval_metrics=retrieval_metrics,
                response_metrics=response_metrics,
                chunking_metrics=chunking_metrics,
                answer=response["answer"],
                retrieved_docs=retrieved_docs_dict,
            )

            results.append(result)

        return results

    def run_ab_test(
        self,
        strategies: list[ChunkingStrategy],
        documents: list[Document],
        test_cases: list[QueryTestCase],
        test_name: str = "chunking_ab_test",
    ) -> ABTestResults:
        """Run A/B test comparing multiple chunking strategies.

        Args:
            strategies: List of chunking strategies to compare.
            documents: Original documents to chunk and test.
            test_cases: List of query test cases.
            test_name: Name for this test run.

        Returns:
            ABTestResults with all test results and summary metrics.
        """
        all_results: dict[str, list[StrategyTestResult]] = {}

        # Test each strategy
        for strategy in strategies:
            print(f"Testing strategy: {strategy.name}")
            results = self.test_strategy(strategy, documents, test_cases)
            all_results[strategy.name] = results

        # Calculate summary metrics
        summary_metrics = self._calculate_summary_metrics(all_results)

        return ABTestResults(
            test_name=test_name,
            strategies=[s.name for s in strategies],
            test_cases=test_cases,
            results=all_results,
            summary_metrics=summary_metrics,
        )

    def _calculate_summary_metrics(
        self, results: dict[str, list[StrategyTestResult]]
    ) -> dict[str, dict[str, float]]:
        """Calculate aggregate metrics across all strategies.

        Args:
            results: Dictionary mapping strategy names to their test results.

        Returns:
            Dictionary mapping strategy names to their summary metrics.
        """
        summary: dict[str, dict[str, float]] = {}

        for strategy_name, strategy_results in results.items():
            if not strategy_results:
                continue

            # Aggregate metrics
            avg_precision = np.mean(
                [
                    r.retrieval_metrics.precision_at_k
                    for r in strategy_results
                ]
            )
            avg_recall = np.mean(
                [r.retrieval_metrics.recall_at_k for r in strategy_results]
            )
            avg_mrr = np.mean(
                [
                    r.retrieval_metrics.mean_reciprocal_rank
                    for r in strategy_results
                ]
            )
            avg_similarity = np.mean(
                [
                    r.retrieval_metrics.mean_similarity_score
                    for r in strategy_results
                ]
            )
            avg_response_time = np.mean(
                [r.response_metrics.response_time_seconds for r in strategy_results]
            )
            avg_semantic_similarity = np.mean(
                [r.response_metrics.semantic_similarity for r in strategy_results]
            )
            avg_chunks = np.mean(
                [r.chunking_metrics.total_chunks for r in strategy_results]
            )
            avg_chunk_size = np.mean(
                [r.chunking_metrics.avg_chunk_size for r in strategy_results]
            )

            summary[strategy_name] = {
                "avg_precision_at_k": float(avg_precision),
                "avg_recall_at_k": float(avg_recall),
                "avg_mrr": float(avg_mrr),
                "avg_retrieval_similarity": float(avg_similarity),
                "avg_response_time_seconds": float(avg_response_time),
                "avg_semantic_similarity": float(avg_semantic_similarity),
                "avg_total_chunks": float(avg_chunks),
                "avg_chunk_size": float(avg_chunk_size),
            }

        return summary

    def save_results(
        self, results: ABTestResults, output_path: str | Path
    ) -> None:
        """Save test results to a JSON file.

        Args:
            results: ABTestResults to save.
            output_path: Path to save the results file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dictionary for JSON serialization
        results_dict = {
            "test_name": results.test_name,
            "strategies": results.strategies,
            "test_cases": [
                {
                    "query": tc.query,
                    "expected_keywords": tc.expected_keywords,
                    "expected_sources": tc.expected_sources,
                    "ground_truth_answer": tc.ground_truth_answer,
                }
                for tc in results.test_cases
            ],
            "summary_metrics": results.summary_metrics,
            "results": {},
            "timestamp": results.timestamp,
        }

        # Serialize results
        for strategy_name, strategy_results in results.results.items():
            results_dict["results"][strategy_name] = []
            for result in strategy_results:
                result_dict = {
                    "strategy_name": result.strategy_name,
                    "query": result.query,
                    "retrieval_metrics": asdict(result.retrieval_metrics),
                    "response_metrics": asdict(result.response_metrics),
                    "chunking_metrics": asdict(result.chunking_metrics),
                    "answer": result.answer,
                    "retrieved_docs": result.retrieved_docs,
                    "timestamp": result.timestamp,
                }
                results_dict["results"][strategy_name].append(result_dict)

        with output_path.open("w") as f:
            json.dump(results_dict, f, indent=2)

    def load_results(self, input_path: str | Path) -> ABTestResults:
        """Load test results from a JSON file.

        Args:
            input_path: Path to load results from.

        Returns:
            ABTestResults loaded from file.
        """
        input_path = Path(input_path)

        with input_path.open() as f:
            data = json.load(f)

        # Reconstruct test cases
        test_cases = [
            QueryTestCase(
                query=tc["query"],
                expected_keywords=tc.get("expected_keywords"),
                expected_sources=tc.get("expected_sources"),
                ground_truth_answer=tc.get("ground_truth_answer"),
            )
            for tc in data["test_cases"]
        ]

        # Reconstruct results
        results: dict[str, list[StrategyTestResult]] = {}
        for strategy_name, strategy_results in data["results"].items():
            results[strategy_name] = []
            for result_data in strategy_results:
                result = StrategyTestResult(
                    strategy_name=result_data["strategy_name"],
                    query=result_data["query"],
                    retrieval_metrics=RetrievalMetrics(
                        **result_data["retrieval_metrics"]
                    ),
                    response_metrics=ResponseMetrics(
                        **result_data["response_metrics"]
                    ),
                    chunking_metrics=ChunkingMetrics(
                        **result_data["chunking_metrics"]
                    ),
                    answer=result_data["answer"],
                    retrieved_docs=result_data["retrieved_docs"],
                    timestamp=result_data.get("timestamp", ""),
                )
                results[strategy_name].append(result)

        return ABTestResults(
            test_name=data["test_name"],
            strategies=data["strategies"],
            test_cases=test_cases,
            results=results,
            summary_metrics=data.get("summary_metrics", {}),
            timestamp=data.get("timestamp", ""),
        )

    def print_comparison_report(self, results: ABTestResults) -> None:
        """Print a formatted comparison report.

        Args:
            results: ABTestResults to print.
        """
        print("\n" + "=" * 80)
        print(f"A/B Test Results: {results.test_name}")
        print("=" * 80)

        print("\nSummary Metrics:")
        print("-" * 80)
        print(
            f"{'Strategy':<20} {'Precision':<12} {'Recall':<12} {'MRR':<12} "
            f"{'Ret. Sim.':<12} {'Resp. Time':<12} {'Sem. Sim.':<12} {'Chunks':<10}"
        )
        print("-" * 80)

        for strategy_name in results.strategies:
            metrics = results.summary_metrics.get(strategy_name, {})
            print(
                f"{strategy_name:<20} "
                f"{metrics.get('avg_precision_at_k', 0.0):<12.4f} "
                f"{metrics.get('avg_recall_at_k', 0.0):<12.4f} "
                f"{metrics.get('avg_mrr', 0.0):<12.4f} "
                f"{metrics.get('avg_retrieval_similarity', 0.0):<12.4f} "
                f"{metrics.get('avg_response_time_seconds', 0.0):<12.4f} "
                f"{metrics.get('avg_semantic_similarity', 0.0):<12.4f} "
                f"{metrics.get('avg_total_chunks', 0.0):<10.0f}"
            )

        print("\n" + "=" * 80)
        print("Detailed Results by Query:")
        print("=" * 80)

        for i, test_case in enumerate(results.test_cases, 1):
            print(f"\nQuery {i}: {test_case.query}")
            print("-" * 80)

            for strategy_name in results.strategies:
                strategy_results = results.results.get(strategy_name, [])
                if i <= len(strategy_results):
                    result = strategy_results[i - 1]
                    print(f"\nStrategy: {strategy_name}")
                    print(
                        f"  Precision@K: {result.retrieval_metrics.precision_at_k:.4f}"
                    )
                    print(
                        f"  Recall@K: {result.retrieval_metrics.recall_at_k:.4f}"
                    )
                    print(f"  MRR: {result.retrieval_metrics.mean_reciprocal_rank:.4f}")
                    print(
                        f"  Avg Similarity: {result.retrieval_metrics.mean_similarity_score:.4f}"
                    )
                    print(
                        f"  Response Time: {result.response_metrics.response_time_seconds:.4f}s"
                    )
                    print(
                        f"  Semantic Similarity: {result.response_metrics.semantic_similarity:.4f}"
                    )
                    print(f"  Answer Length: {result.response_metrics.answer_length}")
                    print(
                        f"  Total Chunks: {result.chunking_metrics.total_chunks}"
                    )
                    print(
                        f"  Avg Chunk Size: {result.chunking_metrics.avg_chunk_size:.0f}"
                    )

        print("\n" + "=" * 80)

