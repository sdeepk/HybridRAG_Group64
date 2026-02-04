import streamlit as st
import time
import sys

sys.path.append(".")

from src.rag.pipeline import run_rag_pipeline

st.set_page_config(page_title="Hybrid RAG System", layout="wide")
st.title("Hybrid RAG System (Dense + BM25 + RRF)")

query = st.text_input("Enter your question")

if st.button("Search") and query:
    with st.spinner("Running Hybrid RAG..."):
        start = time.time()
        result = run_rag_pipeline(query)
        end = time.time()

    st.subheader("Answer")
    st.write(result["answer"])

    st.markdown(f"**Response Time:** {end - start:.2f} seconds")

    st.subheader("Retrieved Context Chunks")

    for i, chunk in enumerate(result["contexts"], 1):
        with st.expander(f"Chunk {i} | RRF Score: {chunk['rrf_score']:.4f}"):
            st.write(chunk["text"])
            st.markdown(f"**Source:** {chunk['url']}")
            st.markdown(
                f"Dense Rank: {chunk['dense_rank']} | "
                f"Sparse Rank: {chunk['sparse_rank']}"
            )
