"""RAG chain implementation with citation support."""

import re
from typing import Any

from langchain_core.callbacks import CallbackManager
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough


class RAGChain:
    """RAG chain that retrieves relevant documents and generates answers with citations."""

    def __init__(
        self,
        llm: BaseChatModel,
        retriever: BaseRetriever,
        top_k: int = 5,
        score_threshold: float = 0.7,
    ) -> None:
        """Initialize the RAG chain.

        Args:
            llm: Language model to use for generation.
            retriever: Retriever to use for document retrieval.
            top_k: Number of documents to retrieve.
            score_threshold: Minimum similarity score threshold for retrieval.
        """
        self.llm = llm
        self.retriever = retriever
        self.top_k = top_k
        self.score_threshold = score_threshold

        # Create the prompt template
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a helpful assistant that answers questions based on the provided context.
Always cite your sources using the source information provided with each context document.
If the context does not contain enough information to answer the question, say so.

Format citations as: [Source: <source_path>]""",
                ),
                (
                    "human",
                    """Context:
{context}

Question: {question}

Answer the question based on the context above. Include citations for your sources.""",
                ),
            ]
        )

        # Build the chain
        self.chain = (
            {
                "context": self._format_documents,
                "question": RunnablePassthrough(),
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        # Create comparison prompt template
        self.comparison_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a helpful assistant that compares and contrasts multiple topics based on provided context documents.
Your goal is to provide a comprehensive, structured comparison that highlights similarities, differences, strengths, and weaknesses.

Always cite your sources using the source information provided with each context document.
If the context does not contain enough information for a thorough comparison, say so.

Format citations as: [Source: <source_path>]

Structure your comparison with:
1. Overview of each topic
2. Key similarities
3. Key differences
4. Advantages/disadvantages of each
5. Use cases or recommendations""",
                ),
                (
                    "human",
                    """Context Documents for Comparison:

{context}

Comparison Request: {question}

Provide a detailed, structured comparison of the topics mentioned above.
Group documents by topic and analyze them side-by-side. Include citations for all sources.""",
                ),
            ]
        )

        # Build comparison chain
        self.comparison_chain = (
            {
                "context": RunnablePassthrough(),
                "question": RunnablePassthrough(),
            }
            | self.comparison_prompt
            | self.llm
            | StrOutputParser()
        )

    def _format_documents(self, question: str) -> str:
        """Retrieve and format documents for the prompt.

        Args:
            question: The question to retrieve documents for.

        Returns:
            Formatted context string with documents and sources.
        """
        docs = self.retriever.invoke(question)

        # Filter by score threshold if similarity_search_with_score is available
        filtered_docs = self._filter_documents_by_score(docs, question)

        formatted_parts: list[str] = []
        for i, doc in enumerate(filtered_docs, 1):
            source = doc.metadata.get("source", "Unknown source")
            chunk_idx = doc.metadata.get("chunk_index", "")
            formatted_parts.append(
                f"[Document {i} - Source: {source}]\n{doc.page_content}\n"
            )

        return "\n---\n\n".join(formatted_parts)

    def _filter_documents_by_score(
        self,
        docs: list[Document],
        question: str,
    ) -> list[Document]:
        """Filter documents by similarity score if the retriever supports it.

        Args:
            docs: List of documents to filter.
            question: The question (used for re-retrieval with scores if needed).

        Returns:
            Filtered list of documents.
        """
        # Try to get scores if the vector store supports it
        try:
            vector_store = self.retriever.vectorstore
            if hasattr(vector_store, "similarity_search_with_score"):
                results_with_scores = vector_store.similarity_search_with_score(
                    question, k=self.top_k
                )
                filtered = [
                    doc
                    for doc, score in results_with_scores
                    if 1 - score >= self.score_threshold  # Convert distance to similarity
                ]
                return filtered if filtered else docs[: self.top_k]
        except Exception:
            pass

        # Fallback to regular retrieval
        return docs[: self.top_k]

    def invoke(
        self, question: str, *, callbacks: list[Any] | None = None
    ) -> dict[str, Any]:
        """Invoke the RAG chain with a question.

        Args:
            question: The question to answer.
            callbacks: Optional list of callback handlers for tracking.

        Returns:
            Dictionary containing the answer, retrieved documents, and usage metadata.
        """
        # Retrieve documents for citation
        docs = self.retriever.invoke(question)
        filtered_docs = self._filter_documents_by_score(docs, question)

        # Generate answer with callbacks
        config: dict[str, Any] = {}
        if callbacks:
            config["callbacks"] = CallbackManager(callbacks)

        result = self.chain.invoke(question, config=config)

        # Extract usage metadata from the result if it's an AIMessage
        usage_metadata: UsageMetadata | None = None
        if isinstance(result, AIMessage) and result.usage_metadata:
            usage_metadata = result.usage_metadata
            answer = result.content if hasattr(result, "content") else str(result)
        else:
            answer = result

        # Extract sources
        sources = [
            {
                "source": doc.metadata.get("source", "Unknown"),
                "chunk_index": doc.metadata.get("chunk_index"),
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            }
            for doc in filtered_docs
        ]

        response: dict[str, Any] = {
            "answer": answer,
            "sources": sources,
            "num_sources": len(sources),
        }

        # Add usage metadata if available
        if usage_metadata:
            response["usage_metadata"] = {
                "input_tokens": usage_metadata.get("input_tokens", 0),
                "output_tokens": usage_metadata.get("output_tokens", 0),
                "total_tokens": usage_metadata.get("total_tokens", 0),
            }

        return response

    async def ainvoke(
        self, question: str, *, callbacks: list[Any] | None = None
    ) -> dict[str, Any]:
        """Asynchronously invoke the RAG chain with a question.

        Args:
            question: The question to answer.
            callbacks: Optional list of callback handlers for tracking.

        Returns:
            Dictionary containing the answer, retrieved documents, and usage metadata.
        """
        # Retrieve documents for citation
        docs = await self.retriever.ainvoke(question)
        filtered_docs = self._filter_documents_by_score(docs, question)

        # Generate answer with callbacks
        config: dict[str, Any] = {}
        if callbacks:
            config["callbacks"] = CallbackManager(callbacks)

        result = await self.chain.ainvoke(question, config=config)

        # Extract usage metadata from the result if it's an AIMessage
        usage_metadata: UsageMetadata | None = None
        if isinstance(result, AIMessage) and result.usage_metadata:
            usage_metadata = result.usage_metadata
            answer = result.content if hasattr(result, "content") else str(result)
        else:
            answer = result

        # Extract sources
        sources = [
            {
                "source": doc.metadata.get("source", "Unknown"),
                "chunk_index": doc.metadata.get("chunk_index"),
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            }
            for doc in filtered_docs
        ]

        response: dict[str, Any] = {
            "answer": answer,
            "sources": sources,
            "num_sources": len(sources),
        }

        # Add usage metadata if available
        if usage_metadata:
            response["usage_metadata"] = {
                "input_tokens": usage_metadata.get("input_tokens", 0),
                "output_tokens": usage_metadata.get("output_tokens", 0),
                "total_tokens": usage_metadata.get("total_tokens", 0),
            }

        return response

    def _extract_comparison_entities(self, query: str) -> list[str] | None:
        """Extract entities from a comparison query.

        Args:
            query: The comparison query (e.g., "Compare AWS Lambda vs Google Cloud Functions").

        Returns:
            List of entity names if this is a comparison query, None otherwise.
        """
        # Patterns for comparison queries
        comparison_patterns = [
            r"compare\s+(.+?)\s+vs\s+(.+)",
            r"compare\s+(.+?)\s+versus\s+(.+)",
            r"compare\s+(.+?)\s+and\s+(.+)",
            r"difference\s+between\s+(.+?)\s+and\s+(.+)",
            r"differences\s+between\s+(.+?)\s+and\s+(.+)",
            r"(.+?)\s+vs\s+(.+?)\s+comparison",
            r"(.+?)\s+versus\s+(.+?)\s+comparison",
        ]

        query_lower = query.lower().strip()

        for pattern in comparison_patterns:
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                entities = [match.group(1).strip(), match.group(2).strip()]
                # Clean up entities (remove trailing punctuation, etc.)
                entities = [re.sub(r"[.,;:!?]+$", "", e).strip() for e in entities]
                return entities if len(entities) == 2 else None

        return None

    def _format_comparison_documents(
        self, entity_docs: dict[str, list[Document]]
    ) -> str:
        """Format documents for comparison by grouping them by entity.

        Args:
            entity_docs: Dictionary mapping entity names to their retrieved documents.

        Returns:
            Formatted context string with documents grouped by entity.
        """
        formatted_parts: list[str] = []

        for entity, docs in entity_docs.items():
            if not docs:
                continue

            filtered_docs = self._filter_documents_by_score(
                docs, f"information about {entity}"
            )

            formatted_parts.append(f"=== DOCUMENTS ABOUT: {entity.upper()} ===\n")
            for i, doc in enumerate(filtered_docs, 1):
                source = doc.metadata.get("source", "Unknown source")
                formatted_parts.append(
                    f"[Document {i} - Source: {source}]\n{doc.page_content}\n"
                )
            formatted_parts.append("\n")

        return "\n---\n\n".join(formatted_parts)

    def compare(
        self,
        query: str,
        entities: list[str] | None = None,
        *,
        callbacks: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Compare multiple entities or topics using retrieved documents.

        Args:
            query: The comparison query (e.g., "Compare AWS Lambda vs Google Cloud Functions").
            entities: Optional list of entities to compare. If None, will attempt to extract
                from the query.
            callbacks: Optional list of callback handlers for tracking.

        Returns:
            Dictionary containing the comparison answer, sources grouped by entity, and metadata.

        Raises:
            ValueError: If entities cannot be extracted and none are provided.
        """
        # Extract entities if not provided
        if entities is None:
            entities = self._extract_comparison_entities(query)
            if entities is None:
                msg = (
                    "Could not extract entities from query. "
                    "Please provide entities explicitly or use a comparison query format "
                    "(e.g., 'Compare X vs Y' or 'Compare X and Y')."
                )
                raise ValueError(msg)

        if len(entities) < 2:
            msg = "Comparison requires at least 2 entities."
            raise ValueError(msg)

        # Retrieve documents for each entity separately
        entity_docs: dict[str, list[Document]] = {}
        all_sources: list[dict[str, Any]] = []

        for entity in entities:
            # Use entity name as query to retrieve relevant documents
            entity_query = f"information about {entity} features capabilities"
            docs = self.retriever.invoke(entity_query)
            entity_docs[entity] = docs

            # Collect sources for this entity
            filtered_docs = self._filter_documents_by_score(docs, entity_query)
            for doc in filtered_docs:
                all_sources.append(
                    {
                        "entity": entity,
                        "source": doc.metadata.get("source", "Unknown"),
                        "chunk_index": doc.metadata.get("chunk_index"),
                        "content": (
                            doc.page_content[:200] + "..."
                            if len(doc.page_content) > 200
                            else doc.page_content
                        ),
                    }
                )

        # Format documents for comparison
        formatted_context = self._format_comparison_documents(entity_docs)

        # Generate comparison with callbacks
        config: dict[str, Any] = {}
        if callbacks:
            config["callbacks"] = CallbackManager(callbacks)

        result = self.comparison_chain.invoke(
            {"context": formatted_context, "question": query}, config=config
        )

        # Extract usage metadata if available
        usage_metadata: UsageMetadata | None = None
        if isinstance(result, AIMessage) and result.usage_metadata:
            usage_metadata = result.usage_metadata
            comparison = result.content if hasattr(result, "content") else str(result)
        else:
            comparison = result

        response: dict[str, Any] = {
            "answer": comparison,
            "sources": all_sources,
            "entities": entities,
            "num_sources": len(all_sources),
            "sources_per_entity": {
                entity: len(docs) for entity, docs in entity_docs.items()
            },
        }

        # Add usage metadata if available
        if usage_metadata:
            response["usage_metadata"] = {
                "input_tokens": usage_metadata.get("input_tokens", 0),
                "output_tokens": usage_metadata.get("output_tokens", 0),
                "total_tokens": usage_metadata.get("total_tokens", 0),
            }

        return response

    async def acompare(
        self,
        query: str,
        entities: list[str] | None = None,
        *,
        callbacks: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Asynchronously compare multiple entities or topics using retrieved documents.

        Args:
            query: The comparison query (e.g., "Compare AWS Lambda vs Google Cloud Functions").
            entities: Optional list of entities to compare. If None, will attempt to extract
                from the query.
            callbacks: Optional list of callback handlers for tracking.

        Returns:
            Dictionary containing the comparison answer, sources grouped by entity, and metadata.

        Raises:
            ValueError: If entities cannot be extracted and none are provided.
        """
        # Extract entities if not provided
        if entities is None:
            entities = self._extract_comparison_entities(query)
            if entities is None:
                msg = (
                    "Could not extract entities from query. "
                    "Please provide entities explicitly or use a comparison query format "
                    "(e.g., 'Compare X vs Y' or 'Compare X and Y')."
                )
                raise ValueError(msg)

        if len(entities) < 2:
            msg = "Comparison requires at least 2 entities."
            raise ValueError(msg)

        # Retrieve documents for each entity separately
        entity_docs: dict[str, list[Document]] = {}
        all_sources: list[dict[str, Any]] = []

        for entity in entities:
            # Use entity name as query to retrieve relevant documents
            entity_query = f"information about {entity} features capabilities"
            docs = await self.retriever.ainvoke(entity_query)
            entity_docs[entity] = docs

            # Collect sources for this entity
            filtered_docs = self._filter_documents_by_score(docs, entity_query)
            for doc in filtered_docs:
                all_sources.append(
                    {
                        "entity": entity,
                        "source": doc.metadata.get("source", "Unknown"),
                        "chunk_index": doc.metadata.get("chunk_index"),
                        "content": (
                            doc.page_content[:200] + "..."
                            if len(doc.page_content) > 200
                            else doc.page_content
                        ),
                    }
                )

        # Format documents for comparison
        formatted_context = self._format_comparison_documents(entity_docs)

        # Generate comparison with callbacks
        config: dict[str, Any] = {}
        if callbacks:
            config["callbacks"] = CallbackManager(callbacks)

        result = await self.comparison_chain.ainvoke(
            {"context": formatted_context, "question": query}, config=config
        )

        # Extract usage metadata if available
        usage_metadata: UsageMetadata | None = None
        if isinstance(result, AIMessage) and result.usage_metadata:
            usage_metadata = result.usage_metadata
            comparison = result.content if hasattr(result, "content") else str(result)
        else:
            comparison = result

        response: dict[str, Any] = {
            "answer": comparison,
            "sources": all_sources,
            "entities": entities,
            "num_sources": len(all_sources),
            "sources_per_entity": {
                entity: len(docs) for entity, docs in entity_docs.items()
            },
        }

        # Add usage metadata if available
        if usage_metadata:
            response["usage_metadata"] = {
                "input_tokens": usage_metadata.get("input_tokens", 0),
                "output_tokens": usage_metadata.get("output_tokens", 0),
                "total_tokens": usage_metadata.get("total_tokens", 0),
            }

        return response

