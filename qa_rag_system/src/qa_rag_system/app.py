"""Main application class for the Q&A RAG system."""

from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from qa_rag_system.config import AppConfig
from qa_rag_system.cost_callback import CostTrackingCallbackHandler
from qa_rag_system.cost_tracker import CostTracker
from qa_rag_system.document_loader import DocumentChunker, DocumentLoader
from qa_rag_system.embeddings import create_embeddings
from qa_rag_system.feedback_analyzer import FeedbackAnalyzer
from qa_rag_system.feedback_store import FeedbackStore
from qa_rag_system.llm import create_llm
from qa_rag_system.rag_chain import RAGChain
from qa_rag_system.vector_store import create_vector_store


class QARAGSystem:
    """Main Q&A RAG system application."""

    def __init__(self, config: AppConfig | None = None) -> None:
        """Initialize the Q&A RAG system.

        Args:
            config: Application configuration. If None, loads from environment.
        """
        self.config = config or AppConfig.from_env()
        self.embeddings = create_embeddings(self.config.embedding)
        self.llm = create_llm(self.config.llm)
        self.vector_store: Any | None = None
        self.rag_chain: RAGChain | None = None
        self.document_chunks: list[Document] = []
        self.bm25_retriever: Any | None = None

        # Initialize feedback system
        self.feedback_store = FeedbackStore(
            storage_path=self.config.feedback.storage_path
        )
        self.feedback_analyzer = FeedbackAnalyzer(
            feedback_store=self.feedback_store,
            learning_rate=self.config.feedback.learning_rate,
            min_samples=self.config.feedback.min_samples,
        )

        # Initialize cost tracking system
        cost_storage_path = getattr(
            self.config.feedback, "cost_storage_path", "./cost_tracking.json"
        )
        self.cost_tracker = CostTracker(storage_path=cost_storage_path)

    def index_documents(
        self,
        paths: list[str],
        chunking_strategy: str = "recursive",
    ) -> None:
        """Index documents from the given paths.

        Args:
            paths: List of file paths, directory paths, or URLs to index.
            chunking_strategy: Chunking strategy ('recursive' or 'token').
        """
        # Load documents
        loader = DocumentLoader()
        documents = loader.load_from_paths(paths)

        if not documents:
            msg = "No documents were loaded. Please check your paths."
            raise ValueError(msg)

        # Chunk documents
        chunker = DocumentChunker(
            chunk_size=self.config.rag.chunk_size,
            chunk_overlap=self.config.rag.chunk_overlap,
            strategy=chunking_strategy,
        )
        chunks = chunker.chunk_documents(documents)
        self.document_chunks = chunks

        # Create vector store with documents
        self.vector_store = create_vector_store(
            config=self.config.vector_db,
            embeddings=self.embeddings,
            documents=chunks,
        )

        # Create retriever
        retriever = self._create_retriever()

        # Create RAG chain
        self.rag_chain = RAGChain(
            llm=self.llm,
            retriever=retriever,
            top_k=self.config.rag.top_k,
            score_threshold=self.config.rag.score_threshold,
        )

    def load_existing_index(self) -> None:
        """Load an existing vector store index."""
        # Create vector store without documents (loads existing)
        self.vector_store = create_vector_store(
            config=self.config.vector_db,
            embeddings=self.embeddings,
            documents=None,
        )

        # Note: For existing indexes, we cannot recreate BM25 retriever
        # without the original documents. Hybrid retrieval will fall back
        # to vector-only retrieval if BM25 is not available.
        if self.config.rag.use_hybrid_retrieval:
            try:
                # Try to create BM25 retriever if we have chunks
                if self.document_chunks:
                    self._create_bm25_retriever()
            except Exception:
                # If BM25 cannot be created, hybrid retrieval will use vector only
                pass

        # Create retriever
        retriever = self._create_retriever()

        # Create RAG chain
        self.rag_chain = RAGChain(
            llm=self.llm,
            retriever=retriever,
            top_k=self.config.rag.top_k,
            score_threshold=self.config.rag.score_threshold,
        )

    def _create_bm25_retriever(self) -> None:
        """Create BM25 retriever from document chunks."""
        try:
            from langchain_community.retrievers import BM25Retriever

            if self.document_chunks:
                self.bm25_retriever = BM25Retriever.from_documents(
                    self.document_chunks
                )
                self.bm25_retriever.k = self.config.rag.top_k
        except ImportError:
            msg = "langchain_community is required for BM25 retrieval. Install it with: pip install langchain-community"
            raise ImportError(msg)

    def _create_retriever(self) -> BaseRetriever:
        """Create retriever based on configuration.

        Returns:
            BaseRetriever instance (either vector-only or hybrid).
        """
        vector_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.config.rag.top_k},
        )

        # Use hybrid retrieval if enabled and BM25 is available
        if self.config.rag.use_hybrid_retrieval:
            # Create BM25 retriever if not already created
            if self.bm25_retriever is None and self.document_chunks:
                self._create_bm25_retriever()

            if self.bm25_retriever is not None:
                try:
                    from langchain_classic.retrievers.hybrid import HybridRetriever
                except ImportError:
                    import warnings

                    warnings.warn(
                        "langchain-classic is required for hybrid retrieval. "
                        "Falling back to vector-only retrieval. "
                        "Install it with: pip install langchain-classic",
                        UserWarning,
                        stacklevel=2,
                    )
                    return vector_retriever

                # Update weights based on feedback if auto-update is enabled
                if self.config.feedback.auto_update_weights:
                    optimal_weights = self.feedback_analyzer.analyze_feedback()
                    semantic_weight = optimal_weights["semantic_weight"]
                    keyword_weight = optimal_weights["keyword_weight"]
                else:
                    semantic_weight = self.config.rag.semantic_weight
                    keyword_weight = self.config.rag.keyword_weight

                return HybridRetriever(
                    vectorstore=self.vector_store,
                    bm25_retriever=self.bm25_retriever,
                    semantic_weight=semantic_weight,
                    keyword_weight=keyword_weight,
                )

        # Fallback to vector-only retrieval
        return vector_retriever

    def update_retrieval_weights(
        self, semantic_weight: float, keyword_weight: float
    ) -> None:
        """Update retrieval weights and recreate retriever.

        Args:
            semantic_weight: New semantic retrieval weight.
            keyword_weight: New keyword retrieval weight.
        """
        # Normalize weights
        total = semantic_weight + keyword_weight
        if total > 0:
            semantic_weight = semantic_weight / total
            keyword_weight = keyword_weight / total

        self.config.rag.semantic_weight = semantic_weight
        self.config.rag.keyword_weight = keyword_weight

        # Recreate retriever with new weights
        if self.rag_chain is not None:
            retriever = self._create_retriever()
            self.rag_chain.retriever = retriever

    def query(self, question: str) -> dict[str, Any]:
        """Query the RAG system with a question.

        Args:
            question: The question to answer.

        Returns:
            Dictionary containing the answer and sources.

        Raises:
            ValueError: If the system has not been initialized with documents.
        """
        if self.rag_chain is None:
            msg = "System not initialized. Please call index_documents() or load_existing_index() first."
            raise ValueError(msg)

        # Create callback handler for cost tracking
        cost_callback = CostTrackingCallbackHandler()

        result = self.rag_chain.invoke(question, callbacks=[cost_callback])

        # Track token usage
        usage_metadata = result.get("usage_metadata")
        if usage_metadata is None:
            # Try to get from callback handler
            callback_usage = cost_callback.get_usage_metadata()
            if callback_usage:
                usage_metadata = {
                    "input_tokens": callback_usage.get("input_tokens", 0),
                    "output_tokens": callback_usage.get("output_tokens", 0),
                    "total_tokens": callback_usage.get("total_tokens", 0),
                }
                result["usage_metadata"] = usage_metadata

        # Track cost
        if usage_metadata:
            model_name = cost_callback.get_model_name() or self.config.llm.model
            self.cost_tracker.track_usage(
                query_type="regular",
                usage_metadata=usage_metadata,
                query_text=question,
                model=model_name,
            )

        # Store query with current weights for feedback tracking
        feedback_id = self.feedback_store.add_feedback(
            query=question,
            answer=result["answer"],
            sources=result["sources"],
            semantic_weight=self.config.rag.semantic_weight,
            keyword_weight=self.config.rag.keyword_weight,
        )

        # Add feedback_id to result for UI
        result["feedback_id"] = feedback_id

        return result

    async def aquery(self, question: str) -> dict[str, Any]:
        """Asynchronously query the RAG system with a question.

        Args:
            question: The question to answer.

        Returns:
            Dictionary containing the answer and sources.

        Raises:
            ValueError: If the system has not been initialized with documents.
        """
        if self.rag_chain is None:
            msg = "System not initialized. Please call index_documents() or load_existing_index() first."
            raise ValueError(msg)

        # Create callback handler for cost tracking
        cost_callback = CostTrackingCallbackHandler()

        result = await self.rag_chain.ainvoke(question, callbacks=[cost_callback])

        # Track token usage
        usage_metadata = result.get("usage_metadata")
        if usage_metadata is None:
            # Try to get from callback handler
            callback_usage = cost_callback.get_usage_metadata()
            if callback_usage:
                usage_metadata = {
                    "input_tokens": callback_usage.get("input_tokens", 0),
                    "output_tokens": callback_usage.get("output_tokens", 0),
                    "total_tokens": callback_usage.get("total_tokens", 0),
                }
                result["usage_metadata"] = usage_metadata

        # Track cost
        if usage_metadata:
            model_name = cost_callback.get_model_name() or self.config.llm.model
            self.cost_tracker.track_usage(
                query_type="regular",
                usage_metadata=usage_metadata,
                query_text=question,
                model=model_name,
            )

        # Store query with current weights for feedback tracking
        feedback_id = self.feedback_store.add_feedback(
            query=question,
            answer=result["answer"],
            sources=result["sources"],
            semantic_weight=self.config.rag.semantic_weight,
            keyword_weight=self.config.rag.keyword_weight,
        )

        # Add feedback_id to result for UI
        result["feedback_id"] = feedback_id

        return result

    def submit_feedback(
        self, feedback_id: str, rating: int, comment: str | None = None
    ) -> bool:
        """Submit feedback for a query.

        Args:
            feedback_id: The feedback ID from the query result.
            rating: User rating (1-5).
            comment: Optional comment from the user.

        Returns:
            True if feedback was submitted successfully.
        """
        success = self.feedback_store.update_feedback(
            feedback_id=feedback_id, rating=rating, comment=comment
        )

        # Auto-update weights if enabled
        if success and self.config.feedback.auto_update_weights:
            optimal_weights = self.feedback_analyzer.analyze_feedback()
            self.update_retrieval_weights(
                semantic_weight=optimal_weights["semantic_weight"],
                keyword_weight=optimal_weights["keyword_weight"],
            )

        return success

    def get_weight_recommendations(self) -> dict[str, Any]:
        """Get weight recommendations based on feedback analysis.

        Returns:
            Dictionary with current weights, recommendations, and statistics.
        """
        return self.feedback_analyzer.get_weight_recommendations(
            current_semantic_weight=self.config.rag.semantic_weight,
            current_keyword_weight=self.config.rag.keyword_weight,
        )

    def compare(
        self, query: str, entities: list[str] | None = None
    ) -> dict[str, Any]:
        """Compare multiple entities or topics using retrieved documents.

        Args:
            query: The comparison query (e.g., "Compare AWS Lambda vs Google Cloud Functions").
            entities: Optional list of entities to compare. If None, will attempt to extract
                from the query.

        Returns:
            Dictionary containing the comparison answer, sources grouped by entity, and metadata.

        Raises:
            ValueError: If the system has not been initialized with documents or entities
                cannot be extracted.
        """
        if self.rag_chain is None:
            msg = "System not initialized. Please call index_documents() or load_existing_index() first."
            raise ValueError(msg)

        # Create callback handler for cost tracking
        cost_callback = CostTrackingCallbackHandler()

        result = self.rag_chain.compare(query, entities, callbacks=[cost_callback])

        # Track token usage
        usage_metadata = result.get("usage_metadata")
        if usage_metadata is None:
            # Try to get from callback handler
            callback_usage = cost_callback.get_usage_metadata()
            if callback_usage:
                usage_metadata = {
                    "input_tokens": callback_usage.get("input_tokens", 0),
                    "output_tokens": callback_usage.get("output_tokens", 0),
                    "total_tokens": callback_usage.get("total_tokens", 0),
                }
                result["usage_metadata"] = usage_metadata

        # Track cost
        if usage_metadata:
            model_name = cost_callback.get_model_name() or self.config.llm.model
            self.cost_tracker.track_usage(
                query_type="comparison",
                usage_metadata=usage_metadata,
                query_text=query,
                model=model_name,
            )

        return result

    async def acompare(
        self, query: str, entities: list[str] | None = None
    ) -> dict[str, Any]:
        """Asynchronously compare multiple entities or topics using retrieved documents.

        Args:
            query: The comparison query (e.g., "Compare AWS Lambda vs Google Cloud Functions").
            entities: Optional list of entities to compare. If None, will attempt to extract
                from the query.

        Returns:
            Dictionary containing the comparison answer, sources grouped by entity, and metadata.

        Raises:
            ValueError: If the system has not been initialized with documents or entities
                cannot be extracted.
        """
        if self.rag_chain is None:
            msg = "System not initialized. Please call index_documents() or load_existing_index() first."
            raise ValueError(msg)

        # Create callback handler for cost tracking
        cost_callback = CostTrackingCallbackHandler()

        result = await self.rag_chain.acompare(query, entities, callbacks=[cost_callback])

        # Track token usage
        usage_metadata = result.get("usage_metadata")
        if usage_metadata is None:
            # Try to get from callback handler
            callback_usage = cost_callback.get_usage_metadata()
            if callback_usage:
                usage_metadata = {
                    "input_tokens": callback_usage.get("input_tokens", 0),
                    "output_tokens": callback_usage.get("output_tokens", 0),
                    "total_tokens": callback_usage.get("total_tokens", 0),
                }
                result["usage_metadata"] = usage_metadata

        # Track cost
        if usage_metadata:
            model_name = cost_callback.get_model_name() or self.config.llm.model
            self.cost_tracker.track_usage(
                query_type="comparison",
                usage_metadata=usage_metadata,
                query_text=query,
                model=model_name,
            )

        return result

    def get_cost_stats(self) -> dict[str, Any]:
        """Get cost tracking statistics.

        Returns:
            Dictionary containing cost statistics by query type.
        """
        return self.cost_tracker.get_all_stats()

    def get_recent_cost_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent cost tracking history.

        Args:
            limit: Maximum number of recent queries to return.

        Returns:
            List of recent query usage dictionaries.
        """
        return self.cost_tracker.get_recent_queries(limit=limit)

