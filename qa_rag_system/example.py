"""Example usage of the Q&A RAG system."""

from qa_rag_system.app import QARAGSystem
from qa_rag_system.config import AppConfig


def main() -> None:
    """Example usage."""
    # Load configuration from environment
    config = AppConfig.from_env()

    # Initialize the system
    rag_system = QARAGSystem(config)

    # Example 1: Index documents from paths
    print("Indexing documents...")
    rag_system.index_documents([
        # Add your document paths here
        # "./documents/python_docs.pdf",
        # "./documents/aws_guide.md",
        # Or use URLs:
        # "https://docs.python.org/3/library/os.html"
    ])

    # Example 2: Query the system
    print("\nQuerying the system...")
    result = rag_system.query("How do I read a file in Python?")

    print("\nAnswer:")
    print(result["answer"])
    print(f"\nSources ({result['num_sources']}):")
    for i, source in enumerate(result["sources"], 1):
        print(f"{i}. {source['source']}")


if __name__ == "__main__":
    main()

