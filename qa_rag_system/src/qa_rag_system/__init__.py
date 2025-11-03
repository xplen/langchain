"""Q&A RAG System for technical documentation with cited sources."""

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

__version__ = "0.1.0"

__all__ = [
    "ABTestResults",
    "ChunkingABTester",
    "ChunkingMetrics",
    "ChunkingStrategy",
    "QueryTestCase",
    "RetrievalMetrics",
    "ResponseMetrics",
    "StrategyTestResult",
]

