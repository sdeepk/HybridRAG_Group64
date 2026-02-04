import streamlit as st
import time
import sys

sys.path.append(".")

from src.rag.pipeline import run_rag_pipeline

# Set page configuration
st.set_page_config(
    page_title="Hybrid RAG System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar
st.sidebar.title("Hybrid RAG System")
st.sidebar.markdown(
    """
    Welcome to the **Hybrid RAG System**! This app combines Dense, BM25, and RRF retrieval methods to provide accurate answers to your questions.

    **Instructions:**
    1. Enter your question in the text box below.
    2. Click on the **Search** button.
    3. View the answer and retrieved context chunks.
    """
)

# Footer inside the sidebar
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <style>
    .footer {
        text-align: center;
        font-size: 14px;
        color: #f1f1f1;
    }
    </style>
    <div class="footer">
        CAI Team 64 | <a href="https://github.com/sdeepk/HybridRAG_Group64" target="_blank">GitHub</a>
    </div>
    """,
    unsafe_allow_html=True,
)

# Main app layout
st.title("📚 Hybrid RAG System (Dense + BM25 + RRF)")
st.markdown("---")

# Input section
query = st.text_input("### Enter your question below:")

# Search button
if st.button("🔍 Search") and query:
    with st.spinner("Running Hybrid RAG... Please wait."):
        start = time.time()
        result = run_rag_pipeline(query)
        end = time.time()

    # Display results
    st.success("Search completed!")
    st.markdown("---")

    # Display answer
    st.subheader("Answer")
    st.write(result["answer"])

    st.markdown(f"**Response Time:** {end - start:.2f} seconds")

    st.markdown("---")

    # Display retrieved context chunks in collapsible sections
    st.subheader("Retrieved Context Chunks")
    for i, chunk in enumerate(result["contexts"], 1):
        with st.expander(f"Chunk {i} | RRF Score: {chunk['rrf_score']:.4f}"):
            st.write(chunk["text"])
            st.markdown(f"**Source:** {chunk['url']}")
            st.markdown(
                f"Dense Rank: {chunk['dense_rank']} | Sparse Rank: {chunk['sparse_rank']}"
            )
