# A/B Testing Framework for Chunking Strategies

This framework allows you to systematically compare different chunking strategies with measurable metrics to determine which approach works best for your RAG system.

## Overview

The A/B testing framework evaluates chunking strategies across multiple dimensions:

- **Retrieval Metrics**: Precision@K, Recall@K, Mean Reciprocal Rank (MRR), similarity scores
- **Response Quality**: Semantic similarity to ground truth, answer length, number of sources cited
- **Performance Metrics**: Response time, chunking statistics (average chunk size, total chunks)

## Quick Start

### Basic Usage

```python
from qa_rag_system.ab_testing import (
    ChunkingABTester,
    ChunkingStrategy,
    QueryTestCase,
)
from qa_rag_system.config import AppConfig
from qa_rag_system.embeddings import create_embeddings
from qa_rag_system.llm import create_llm
from qa_rag_system.document_loader import DocumentLoader

# Load configuration
config = AppConfig.from_env()

# Initialize components
embeddings = create_embeddings(config.embedding)
llm = create_llm(config.llm)

# Create A/B tester
tester = ChunkingABTester(
    embeddings=embeddings,
    llm=llm,
    vector_store_config=config.vector_db,
    top_k=config.rag.top_k,
    score_threshold=config.rag.score_threshold,
)

# Define strategies to test
strategies = [
    ChunkingStrategy(
        name="small_chunks",
        strategy_type="recursive",
        chunk_size=500,
        chunk_overlap=100,
    ),
    ChunkingStrategy(
        name="medium_chunks",
        strategy_type="recursive",
        chunk_size=1000,
        chunk_overlap=200,
    ),
    ChunkingStrategy(
        name="large_chunks",
        strategy_type="recursive",
        chunk_size=2000,
        chunk_overlap=400,
    ),
]

# Load documents
loader = DocumentLoader()
documents = loader.load_from_paths(["./docs"])

# Define test queries
test_cases = [
    QueryTestCase(
        query="What are the main features?",
        expected_keywords=["feature", "capability"],
        expected_sources=["docs/features.md"],
    ),
    QueryTestCase(
        query="How do I install?",
        expected_keywords=["install", "setup"],
    ),
]

# Run A/B test
results = tester.run_ab_test(
    strategies=strategies,
    documents=documents,
    test_cases=test_cases,
    test_name="chunk_size_comparison",
)

# Print comparison report
tester.print_comparison_report(results)

# Save results
tester.save_results(results, "./ab_test_results.json")
```

## Chunking Strategies

### Supported Strategy Types

1. **`recursive`**: RecursiveCharacterTextSplitter
   - Splits text recursively by paragraphs, sentences, and words
   - Best for structured documents
   - Preserves document structure

2. **`token`**: TokenTextSplitter
   - Splits by token count
   - Useful for LLM token limits
   - Best for code or unstructured text

### Strategy Configuration

```python
ChunkingStrategy(
    name="strategy_name",        # Unique identifier
    strategy_type="recursive",   # 'recursive' or 'token'
    chunk_size=1000,             # Size of each chunk
    chunk_overlap=200,           # Overlap between chunks
    kwargs={},                    # Additional strategy-specific kwargs
)
```

## Test Cases

Test cases define the queries to evaluate and optionally include ground truth information:

```python
QueryTestCase(
    query="Your question here",
    expected_keywords=["keyword1", "keyword2"],  # Optional: for precision calculation
    expected_sources=["source1.txt"],            # Optional: for recall calculation
    ground_truth_answer="Expected answer",        # Optional: for semantic similarity
)
```

## Metrics Explained

### Retrieval Metrics

- **Precision@K**: Fraction of retrieved documents that are relevant
- **Recall@K**: Fraction of relevant documents that were retrieved
- **Mean Reciprocal Rank (MRR)**: Average of reciprocal ranks of first relevant document
- **Mean Similarity Score**: Average similarity score of retrieved documents

### Response Metrics

- **Answer Length**: Character count of generated answer
- **Num Sources Cited**: Number of sources referenced in answer
- **Semantic Similarity**: Cosine similarity between answer and ground truth (if provided)
- **Response Time**: Time taken to generate response

### Chunking Metrics

- **Total Chunks**: Total number of chunks created
- **Avg Chunk Size**: Average character count per chunk
- **Median Chunk Size**: Median character count per chunk
- **Min/Max Chunk Size**: Minimum and maximum chunk sizes
- **Chunk Size Std**: Standard deviation of chunk sizes

## Results Analysis

### Loading Saved Results

```python
# Load previously saved results
results = tester.load_results("./ab_test_results.json")

# Access summary metrics
for strategy_name, metrics in results.summary_metrics.items():
    print(f"{strategy_name}:")
    print(f"  Precision: {metrics['avg_precision_at_k']:.4f}")
    print(f"  Recall: {metrics['avg_recall_at_k']:.4f}")
    print(f"  MRR: {metrics['avg_mrr']:.4f}")
```

### Accessing Detailed Results

```python
# Access results for a specific strategy
strategy_results = results.results["strategy_name"]

for result in strategy_results:
    print(f"Query: {result.query}")
    print(f"Answer: {result.answer}")
    print(f"Precision: {result.retrieval_metrics.precision_at_k}")
    print(f"Response Time: {result.response_metrics.response_time_seconds}s")
```

## Best Practices

1. **Use Representative Test Cases**: Include queries that reflect real user questions
2. **Include Ground Truth**: Provide ground truth answers when possible for semantic similarity metrics
3. **Test Multiple Scenarios**: Vary chunk sizes, overlaps, and strategies
4. **Run Multiple Iterations**: Consider running tests multiple times and averaging results
5. **Consider Your Use Case**: Different metrics matter for different applications:
   - High precision: Focus on Precision@K
   - Comprehensive coverage: Focus on Recall@K
   - Fast responses: Focus on Response Time
   - Quality answers: Focus on Semantic Similarity

## Example: Comparing Chunk Sizes

```python
strategies = [
    ChunkingStrategy(name="small", strategy_type="recursive", chunk_size=500, chunk_overlap=100),
    ChunkingStrategy(name="medium", strategy_type="recursive", chunk_size=1000, chunk_overlap=200),
    ChunkingStrategy(name="large", strategy_type="recursive", chunk_size=2000, chunk_overlap=400),
]

results = tester.run_ab_test(strategies, documents, test_cases)
tester.print_comparison_report(results)
```

## Example: Comparing Strategy Types

```python
strategies = [
    ChunkingStrategy(name="recursive", strategy_type="recursive", chunk_size=1000, chunk_overlap=200),
    ChunkingStrategy(name="token", strategy_type="token", chunk_size=1000, chunk_overlap=200),
]

results = tester.run_ab_test(strategies, documents, test_cases)
tester.print_comparison_report(results)
```

## Limitations

- Each strategy creates its own vector store, which requires additional storage
- Tests can be time-consuming with large document sets
- Semantic similarity requires ground truth answers
- Precision/Recall calculations require expected keywords or sources

## Troubleshooting

**Issue**: Tests take too long
- **Solution**: Use smaller document sets or fewer test cases for initial testing

**Issue**: Low precision/recall scores
- **Solution**: Ensure expected_keywords and expected_sources are accurate and comprehensive

**Issue**: Vector store conflicts
- **Solution**: The framework automatically uses unique collection names per strategy, but ensure no conflicts in persist directories

## See Also

- `examples/ab_testing_example.py` - Complete working example
- `tests/test_ab_testing.py` - Unit tests demonstrating usage

