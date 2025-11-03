"""Streamlit frontend for the Q&A RAG system."""

import os
import tempfile
from pathlib import Path

import streamlit as st

from qa_rag_system.app import QARAGSystem
from qa_rag_system.config import AppConfig

# Page configuration
st.set_page_config(
    page_title="Q&A RAG System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "rag_system" not in st.session_state:
    st.session_state.rag_system = None
if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "show_cost_dashboard" not in st.session_state:
    st.session_state.show_cost_dashboard = False


def main() -> None:
    """Main Streamlit application."""
    st.title("📚 Q&A RAG System")
    st.markdown(
        "Ask questions about your technical documentation with cited sources using RAG architecture."
    )

    # Sidebar for configuration and indexing
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Display current configuration
        st.subheader("Current Settings")
        try:
            config = AppConfig.from_env()
            st.write(f"**LLM:** {config.llm.provider} ({config.llm.model})")
            st.write(f"**Embeddings:** {config.embedding.provider} ({config.embedding.model})")
            st.write(f"**Vector DB:** {config.vector_db.provider}")
            st.write(f"**Chunk Size:** {config.rag.chunk_size}")
            st.write(f"**Top K:** {config.rag.top_k}")
            if config.rag.use_hybrid_retrieval:
                st.write(f"**Hybrid Retrieval:** Enabled")
                st.write(f"**Semantic Weight:** {config.rag.semantic_weight:.2f}")
                st.write(f"**Keyword Weight:** {config.rag.keyword_weight:.2f}")
        except Exception as e:
            st.error(f"Configuration error: {e}")
            st.info("Please set up your .env file with the required configuration.")
            return

        st.divider()

        # Document indexing section
        st.subheader("📄 Index Documents")

        indexing_method = st.radio(
            "Indexing Method",
            ["Upload Files", "Enter Paths", "Load Existing Index"],
            help="Choose how to provide documents for indexing",
        )

        if indexing_method == "Upload Files":
            uploaded_files = st.file_uploader(
                "Upload documents",
                type=["pdf", "txt", "md", "py", "js", "ts", "html"],
                accept_multiple_files=True,
            )

            if uploaded_files and st.button("Index Uploaded Files"):
                with st.spinner("Indexing documents..."):
                    try:
                        # Save uploaded files temporarily
                        temp_dir = tempfile.mkdtemp()
                        file_paths = []

                        for uploaded_file in uploaded_files:
                            file_path = os.path.join(temp_dir, uploaded_file.name)
                            with open(file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            file_paths.append(file_path)

                        # Initialize and index
                        rag_system = QARAGSystem(config)
                        rag_system.index_documents(file_paths)

                        st.session_state.rag_system = rag_system
                        st.session_state.indexed = True
                        st.success(f"Successfully indexed {len(uploaded_files)} file(s)!")
                    except Exception as e:
                        st.error(f"Error indexing documents: {e}")

        elif indexing_method == "Enter Paths":
            paths_input = st.text_area(
                "Enter paths (one per line)",
                help="Enter file paths, directory paths, or URLs, one per line",
                height=100,
            )

            if paths_input and st.button("Index from Paths"):
                with st.spinner("Indexing documents..."):
                    try:
                        paths = [p.strip() for p in paths_input.split("\n") if p.strip()]
                        rag_system = QARAGSystem(config)
                        rag_system.index_documents(paths)

                        st.session_state.rag_system = rag_system
                        st.session_state.indexed = True
                        st.success(f"Successfully indexed {len(paths)} path(s)!")
                    except Exception as e:
                        st.error(f"Error indexing documents: {e}")

        elif indexing_method == "Load Existing Index":
            if st.button("Load Existing Index"):
                with st.spinner("Loading existing index..."):
                    try:
                        rag_system = QARAGSystem(config)
                        rag_system.load_existing_index()

                        st.session_state.rag_system = rag_system
                        st.session_state.indexed = True
                        st.success("Successfully loaded existing index!")
                    except Exception as e:
                        st.error(f"Error loading index: {e}")

        # Display indexing status
        if st.session_state.indexed:
            st.success("✅ Index ready for queries")

        # Cost tracking dashboard section
        if st.session_state.indexed and st.session_state.rag_system:
            st.divider()
            st.subheader("💰 Cost Tracking")

            if st.button("View Cost Dashboard"):
                st.session_state.show_cost_dashboard = True

            if st.session_state.get("show_cost_dashboard", False):
                try:
                    cost_stats = st.session_state.rag_system.get_cost_stats()

                    # Overall statistics
                    overall = cost_stats.get("overall", {})
                    st.write("**Overall Statistics:**")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Queries", overall.get("total_queries", 0))
                    with col2:
                        st.metric("Total Tokens", f"{overall.get('total_tokens', 0):,}")
                    with col3:
                        st.metric("Avg Tokens/Query", f"{overall.get('avg_total_tokens', 0):.1f}")
                    with col4:
                        st.metric("Total Input Tokens", f"{overall.get('total_input_tokens', 0):,}")

                    # Statistics by query type
                    by_type = cost_stats.get("by_query_type", {})
                    if by_type:
                        st.write("**Statistics by Query Type:**")
                        for query_type, stats in by_type.items():
                            with st.expander(f"📊 {query_type.title()} Queries"):
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Total Queries", stats.get("total_queries", 0))
                                with col2:
                                    st.metric("Total Tokens", f"{stats.get('total_tokens', 0):,}")
                                with col3:
                                    st.metric("Avg Tokens", f"{stats.get('avg_total_tokens', 0):.1f}")
                                with col4:
                                    st.metric("Total Output Tokens", f"{stats.get('total_output_tokens', 0):,}")

                                # Breakdown
                                st.write("**Token Breakdown:**")
                                st.write(f"- Input Tokens: {stats.get('total_input_tokens', 0):,} (avg: {stats.get('avg_input_tokens', 0):.1f})")
                                st.write(f"- Output Tokens: {stats.get('total_output_tokens', 0):,} (avg: {stats.get('avg_output_tokens', 0):.1f})")

                        # Visualizations
                        st.write("**Visualizations:**")
                        query_types = list(by_type.keys())
                        total_tokens_by_type = [by_type[qt]["total_tokens"] for qt in query_types]
                        total_queries_by_type = [by_type[qt]["total_queries"] for qt in query_types]

                        col1, col2 = st.columns(2)
                        with col1:
                            st.bar_chart(dict(zip(query_types, total_tokens_by_type)))
                            st.caption("Total Tokens by Query Type")
                        with col2:
                            st.bar_chart(dict(zip(query_types, total_queries_by_type)))
                            st.caption("Total Queries by Query Type")

                    # Recent queries
                    st.write("**Recent Queries:**")
                    recent_history = st.session_state.rag_system.get_recent_cost_history(limit=20)
                    if recent_history:
                        for query_info in recent_history[:10]:  # Show last 10
                            with st.expander(f"{query_info.get('query_type', 'unknown').title()} - {query_info.get('timestamp', '')[:19]}"):
                                st.write(f"**Query:** {query_info.get('query_text', 'N/A')[:100]}...")
                                st.write(f"**Tokens:** {query_info.get('total_tokens', 0):,} (Input: {query_info.get('input_tokens', 0):,}, Output: {query_info.get('output_tokens', 0):,})")
                                if query_info.get('model'):
                                    st.write(f"**Model:** {query_info['model']}")
                    else:
                        st.info("No query history yet. Start asking questions to see usage data.")

                    if st.button("Close Dashboard"):
                        st.session_state.show_cost_dashboard = False
                        st.rerun()

                except Exception as e:
                    st.error(f"Error loading cost dashboard: {e}")

        # Feedback and weight management section
        if st.session_state.indexed and st.session_state.rag_system:
            st.divider()
            st.subheader("🔄 Feedback & Learning")

            # Show feedback statistics
            stats = st.session_state.rag_system.feedback_store.get_statistics()
            st.write(f"**Total Queries:** {stats['total_queries']}")
            st.write(f"**Rated Queries:** {stats['rated_queries']}")
            if stats['rated_queries'] > 0:
                st.write(f"**Average Rating:** {stats['average_rating']:.2f}/5.0")
                st.write("**Rating Distribution:**")
                for rating, count in sorted(stats['rating_distribution'].items()):
                    st.write(f"  {rating}⭐: {count}")

            # Weight recommendations
            if stats['rated_queries'] >= st.session_state.rag_system.config.feedback.min_samples:
                with st.expander("📊 Weight Recommendations"):
                    recommendations = st.session_state.rag_system.get_weight_recommendations()
                    st.write("**Current Weights:**")
                    st.write(f"- Semantic: {recommendations['current_weights']['semantic_weight']:.3f}")
                    st.write(f"- Keyword: {recommendations['current_weights']['keyword_weight']:.3f}")
                    st.write("**Recommended Weights:**")
                    st.write(f"- Semantic: {recommendations['recommended_weights']['semantic_weight']:.3f}")
                    st.write(f"- Keyword: {recommendations['recommended_weights']['keyword_weight']:.3f}")
                    st.write("**Recommended Changes:**")
                    st.write(f"- Semantic: {recommendations['recommended_changes']['semantic_weight']:+.3f}")
                    st.write(f"- Keyword: {recommendations['recommended_changes']['keyword_weight']:+.3f}")

                    if st.button("Apply Recommended Weights"):
                        st.session_state.rag_system.update_retrieval_weights(
                            semantic_weight=recommendations['recommended_weights']['semantic_weight'],
                            keyword_weight=recommendations['recommended_weights']['keyword_weight'],
                        )
                        st.success("Weights updated! New queries will use the optimized weights.")
                        st.rerun()

            # Manual weight adjustment
            with st.expander("⚙️ Manual Weight Adjustment"):
                current_semantic = st.session_state.rag_system.config.rag.semantic_weight
                current_keyword = st.session_state.rag_system.config.rag.keyword_weight

                semantic_weight = st.slider(
                    "Semantic Weight",
                    min_value=0.0,
                    max_value=1.0,
                    value=current_semantic,
                    step=0.05,
                    help="Weight for semantic/vector search",
                )
                keyword_weight = st.slider(
                    "Keyword Weight",
                    min_value=0.0,
                    max_value=1.0,
                    value=current_keyword,
                    step=0.05,
                    help="Weight for keyword/BM25 search",
                )

                # Normalize weights
                total = semantic_weight + keyword_weight
                if total > 0:
                    semantic_weight = semantic_weight / total
                    keyword_weight = keyword_weight / total

                st.write(f"**Normalized Weights:**")
                st.write(f"- Semantic: {semantic_weight:.3f}")
                st.write(f"- Keyword: {keyword_weight:.3f}")

                if st.button("Update Weights"):
                    st.session_state.rag_system.update_retrieval_weights(
                        semantic_weight=semantic_weight,
                        keyword_weight=keyword_weight,
                    )
                    st.success("Weights updated!")
                    st.rerun()

    # Main content area
    if not st.session_state.indexed:
        st.info(
            "👈 Please index documents using the sidebar to start asking questions."
        )
        st.markdown(
            """
        ### Getting Started

        1. **Configure your environment**: Set up your `.env` file with API keys and configuration
        2. **Index documents**: Use the sidebar to upload files, enter paths, or load an existing index
        3. **Ask questions**: Once indexed, you can ask questions about your documentation

        ### Supported Document Types
        - PDF files (`.pdf`)
        - Text files (`.txt`, `.md`)
        - Code files (`.py`, `.js`, `.ts`)
        - HTML files (`.html`)
        - URLs (web pages)

        ### Features
        - **Multiple LLM providers**: OpenAI or Ollama (local)
        - **Multiple embedding models**: OpenAI or Sentence Transformers
        - **Multiple vector databases**: Pinecone or Chroma
        - **Source citations**: Answers include citations to source documents
        - **Flexible chunking**: Multiple chunking strategies
        - **Hybrid retrieval**: Combines semantic and keyword search
        - **Feedback loop**: Rate answers to improve retrieval weights
        """
        )
    else:
        # Query interface
        st.header("💬 Ask a Question")

        question = st.text_input(
            "Enter your question",
            placeholder="e.g., How do I use the authentication API?",
        )

        if question:
            if st.button("Ask", type="primary") or question:
                with st.spinner("Thinking..."):
                    try:
                        result = st.session_state.rag_system.query(question)

                        # Store feedback_id in session state for this query
                        st.session_state.current_feedback_id = result.get("feedback_id")

                        # Display answer
                        st.subheader("Answer")
                        st.markdown(result["answer"])

                        # Display token usage if available
                        if "usage_metadata" in result:
                            usage = result["usage_metadata"]
                            st.caption(
                                f"Token Usage: {usage.get('total_tokens', 0):,} total "
                                f"({usage.get('input_tokens', 0):,} input, "
                                f"{usage.get('output_tokens', 0):,} output)"
                            )

                        # Display sources
                        st.subheader(f"Sources ({result['num_sources']})")
                        for i, source in enumerate(result["sources"], 1):
                            with st.expander(f"Source {i}: {Path(source['source']).name}"):
                                st.write(f"**Path:** {source['source']}")
                                if source.get("chunk_index") is not None:
                                    st.write(f"**Chunk Index:** {source['chunk_index']}")
                                st.write("**Content Preview:**")
                                st.code(source["content"], language=None)

                        # Feedback section
                        st.divider()
                        st.subheader("💬 Rate this Answer")

                        # Check if feedback already submitted
                        feedback_id = result.get("feedback_id")
                        if feedback_id:
                            existing_feedback = st.session_state.rag_system.feedback_store.get_feedback(
                                feedback_id
                            )

                            if existing_feedback and existing_feedback.rating:
                                st.info(
                                    f"✅ You rated this answer {existing_feedback.rating}⭐"
                                    + (f": {existing_feedback.comment}" if existing_feedback.comment else "")
                                )
                            else:
                                col1, col2 = st.columns([2, 1])

                                with col1:
                                    rating = st.slider(
                                        "Rate this answer (1-5)",
                                        min_value=1,
                                        max_value=5,
                                        value=3,
                                        step=1,
                                        help="1 = Poor, 5 = Excellent",
                                    )

                                    comment = st.text_area(
                                        "Optional comment",
                                        placeholder="Tell us what was helpful or what could be improved...",
                                        height=100,
                                    )

                                with col2:
                                    st.write("**Rating Guide:**")
                                    st.write("5⭐ = Excellent, very helpful")
                                    st.write("4⭐ = Good, mostly helpful")
                                    st.write("3⭐ = Average, somewhat helpful")
                                    st.write("2⭐ = Poor, not very helpful")
                                    st.write("1⭐ = Very poor, not helpful")

                                if st.button("Submit Feedback", type="primary"):
                                    success = st.session_state.rag_system.submit_feedback(
                                        feedback_id=feedback_id,
                                        rating=rating,
                                        comment=comment if comment.strip() else None,
                                    )

                                    if success:
                                        st.success(
                                            "Thank you for your feedback! "
                                            "Your rating helps improve the system."
                                        )
                                        if st.session_state.rag_system.config.feedback.auto_update_weights:
                                            st.info(
                                                "Weights have been automatically updated based on your feedback."
                                            )
                                        st.rerun()
                                    else:
                                        st.error("Failed to submit feedback. Please try again.")

                    except Exception as e:
                        st.error(f"Error processing question: {e}")


if __name__ == "__main__":
    main()

