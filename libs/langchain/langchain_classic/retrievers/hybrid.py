"""Hybrid Retriever that combines semantic and keyword (BM25) retrieval."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from pydantic import Field, PrivateAttr, model_validator
from typing_extensions import override

from langchain_classic.retrievers.ensemble import EnsembleRetriever

if TYPE_CHECKING:
    from langchain_community.retrievers import BM25Retriever


class HybridRetriever(BaseRetriever):
    """Retriever that combines semantic (vector) and keyword (BM25) retrieval.

    This retriever combines the strengths of semantic search (which understands
    context and meaning) with keyword-based BM25 search (which matches exact terms).
    It uses Reciprocal Rank Fusion (RRF) to merge results from both retrievers.

    Args:
        vectorstore: VectorStore instance for semantic similarity search.
        bm25_retriever: BM25Retriever instance for keyword-based search.
        semantic_weight: Weight for semantic search results. Defaults to 0.5.
        keyword_weight: Weight for BM25 keyword search results. Defaults to 0.5.
        c: Constant added to the rank in RRF, controlling the balance between
            high-ranked and lower-ranked items. Defaults to 60.
        id_key: Key in document metadata used to determine unique documents.
            If not specified, page_content is used. Defaults to None.

    Example:
        ```python
        from langchain_community.retrievers import BM25Retriever
        from langchain_core.documents import Document
        from langchain_core.vectorstores import VectorStore
        from langchain_openai import OpenAIEmbeddings

        # Create documents
        documents = [
            Document(page_content="Python is a programming language"),
            Document(page_content="Machine learning uses algorithms"),
        ]

        # Create vector store for semantic search
        vectorstore = Chroma.from_documents(
            documents, embedding=OpenAIEmbeddings()
        )

        # Create BM25 retriever for keyword search
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 4

        # Create hybrid retriever
        hybrid_retriever = HybridRetriever(
            vectorstore=vectorstore,
            bm25_retriever=bm25_retriever,
            semantic_weight=0.6,
            keyword_weight=0.4,
        )

        # Use the hybrid retriever
        results = hybrid_retriever.invoke("programming")
        ```
    """

    vectorstore: VectorStore
    """VectorStore instance for semantic similarity search."""
    bm25_retriever: Any = Field(description="BM25Retriever instance for keyword search")
    semantic_weight: float = Field(
        default=0.5, description="Weight for semantic search results"
    )
    keyword_weight: float = Field(
        default=0.5, description="Weight for BM25 keyword search results"
    )
    c: int = Field(
        default=60,
        description=(
            "Constant added to the rank in RRF, controlling the balance "
            "between high-ranked and lower-ranked items"
        ),
    )
    id_key: str | None = Field(
        default=None,
        description=(
            "Key in document metadata used to determine unique documents. "
            "If not specified, page_content is used."
        ),
    )

    _ensemble_retriever: EnsembleRetriever | None = PrivateAttr(default=None)
    """Internal ensemble retriever that combines vector and BM25 retrievers."""

    @model_validator(mode="after")
    def _initialize_ensemble(self) -> HybridRetriever:
        """Initialize the ensemble retriever with vector and BM25 retrievers.

        Returns:
            Self with initialized ensemble retriever.
        """
        # Validate bm25_retriever is a BaseRetriever
        from langchain_core.retrievers import BaseRetriever

        if not isinstance(self.bm25_retriever, BaseRetriever):
            msg = (
                f"bm25_retriever must be an instance of BaseRetriever, "
                f"got {type(self.bm25_retriever)}"
            )
            raise TypeError(msg)

        # Try to validate it's a BM25Retriever if langchain_community is available
        # This is a soft check - we allow any BaseRetriever for flexibility
        try:
            from langchain_community.retrievers import BM25Retriever

            if not isinstance(self.bm25_retriever, BM25Retriever):
                # Warn but don't fail - allows for testing with mocks
                import warnings

                warnings.warn(
                    f"bm25_retriever should be an instance of BM25Retriever "
                    f"from langchain_community, got {type(self.bm25_retriever)}. "
                    "Proceeding anyway, but results may be unexpected.",
                    UserWarning,
                    stacklevel=2,
                )
        except ImportError:
            # langchain_community not available - proceed with BaseRetriever check only
            pass

        # Create vector store retriever for semantic search
        vector_retriever = self.vectorstore.as_retriever()

        # Create ensemble retriever that combines both
        self._ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, self.bm25_retriever],
            weights=[self.semantic_weight, self.keyword_weight],
            c=self.c,
            id_key=self.id_key,
        )

        return self

    @property
    def config_specs(self) -> list:
        """List configurable fields for this runnable."""
        if self._ensemble_retriever is None:
            return []
        return self._ensemble_retriever.config_specs

    @override
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """Get relevant documents using hybrid search.

        Args:
            query: Query string to search for.
            run_manager: Callback manager for this retrieval run.

        Returns:
            List of relevant documents ranked by hybrid search scores.
        """
        if self._ensemble_retriever is None:
            msg = "Ensemble retriever not initialized. This should not happen."
            raise RuntimeError(msg)

        return self._ensemble_retriever._get_relevant_documents(
            query, run_manager=run_manager
        )

    @override
    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """Asynchronously get relevant documents using hybrid search.

        Args:
            query: Query string to search for.
            run_manager: Async callback manager for this retrieval run.

        Returns:
            List of relevant documents ranked by hybrid search scores.
        """
        if self._ensemble_retriever is None:
            msg = "Ensemble retriever not initialized. This should not happen."
            raise RuntimeError(msg)

        return await self._ensemble_retriever._aget_relevant_documents(
            query, run_manager=run_manager
        )

