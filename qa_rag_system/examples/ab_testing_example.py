"""Example script demonstrating A/B testing of chunking strategies."""

import os
from pathlib import Path

from dotenv import load_dotenv

from qa_rag_system.ab_testing import (
    ABTestResults,
    ChunkingABTester,
    ChunkingStrategy,
    QueryTestCase,
)
from qa_rag_system.app import QARAGSystem
from qa_rag_system.config import AppConfig

load_dotenv()


def main() -> None:
    """Run an A/B test comparing different chunking strategies."""
    # Load configuration
    config = AppConfig.from_env()

    # Initialize the A/B tester
    from qa_rag_system.embeddings import create_embeddings
    from qa_rag_system.llm import create_llm

    embeddings = create_embeddings(config.embedding)
    llm = create_llm(config.llm)

    tester = ChunkingABTester(
        embeddings=embeddings,
        llm=llm,
        vector_store_config=config.vector_db,
        top_k=config.rag.top_k,
        score_threshold=config.rag.score_threshold,
    )

    # Define chunking strategies to test
    strategies = [
        ChunkingStrategy(
            name="recursive_small",
            strategy_type="recursive",
            chunk_size=500,
            chunk_overlap=100,
        ),
        ChunkingStrategy(
            name="recursive_medium",
            strategy_type="recursive",
            chunk_size=1000,
            chunk_overlap=200,
        ),
        ChunkingStrategy(
            name="recursive_large",
            strategy_type="recursive",
            chunk_size=2000,
            chunk_overlap=400,
        ),
        ChunkingStrategy(
            name="token_medium",
            strategy_type="token",
            chunk_size=1000,
            chunk_overlap=200,
        ),
    ]

    # Load test documents
    # Replace with your document paths
    doc_paths = []
    if os.path.exists("./docs"):
        doc_paths = ["./docs"]
    elif os.path.exists("../docs"):
        doc_paths = ["../docs"]
    else:
        print("No documents found. Please specify document paths.")
        return

    from qa_rag_system.document_loader import DocumentLoader

    loader = DocumentLoader()
    documents = loader.load_from_paths(doc_paths)

    if not documents:
        print("No documents were loaded.")
        return

    print(f"Loaded {len(documents)} documents")

    # Define test queries
    test_cases = [
        QueryTestCase(
            query="What are the main features?",
            expected_keywords=["feature", "capability", "function"],
        ),
        QueryTestCase(
            query="How do I get started?",
            expected_keywords=["start", "begin", "setup", "install"],
        ),
        QueryTestCase(
            query="What are the configuration options?",
            expected_keywords=["config", "setting", "option", "parameter"],
        ),
    ]

    # Run A/B test
    print("\nRunning A/B test...")
    results = tester.run_ab_test(
        strategies=strategies,
        documents=documents,
        test_cases=test_cases,
        test_name="chunking_strategy_comparison",
    )

    # Print comparison report
    tester.print_comparison_report(results)

    # Save results
    output_path = Path("./ab_test_results.json")
    tester.save_results(results, output_path)
    print(f"\nResults saved to {output_path}")

    # Example: Load and analyze saved results
    print("\nLoading saved results...")
    loaded_results = tester.load_results(output_path)
    print(f"Loaded test: {loaded_results.test_name}")
    print(f"Strategies tested: {', '.join(loaded_results.strategies)}")


if __name__ == "__main__":
    main()

