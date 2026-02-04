# src/rag/pipeline.py

import time
from typing import Dict, List


from src.index.dense_chroma import dense_retrieve
from src.index.sparse_bm25 import sparse_retrieve
from src.rag.rrf import rrf_fusion
from src.rag.generate_flan_t5 import generate_answer_flan_t5

# -----------------------------
# Main RAG pipeline
# -----------------------------

def run_rag_pipeline(
    query: str,
    top_k_dense: int = 50,
    top_k_sparse: int = 50,
    top_n_context: int = 10
) -> Dict:
    """
    Runs Hybrid RAG pipeline:
    Dense + BM25 -> RRF -> LLM generation
    """

    start_time = time.time()

    # Dense retrieval
    dense_results = dense_retrieve(query, top_k=top_k_dense)

    # Sparse (BM25) retrieval
    sparse_results = sparse_retrieve(query, top_k=top_k_sparse)

    # RRF fusion
    fused_chunks = rrf_fusion(
        dense_results,
        sparse_results,
        k=60,
        top_n=top_n_context
    )

    # Prepare contexts for LLM (truncate to avoid overflow)
    contexts = [c["text"][:900] for c in fused_chunks]

    # Generate answer
    answer, gen_debug = generate_answer_flan_t5(
        query=query,
        contexts=contexts,
        max_input_tokens=1536,
        max_new_tokens=220
    )

    end_time = time.time()

    # Format chunks for UI
    ui_chunks = []
    for c in fused_chunks:
        ui_chunks.append({
            "text": c["text"],
            "url": c["url"],
            "title": c.get("title", ""),
            "dense_rank": c.get("dense_rank"),
            "sparse_rank": c.get("sparse_rank"),
            "rrf_score": c.get("rrf_score")
        })

    return {
        "query": query,
        "answer": answer,
        "contexts": ui_chunks,
        "response_time": round(end_time - start_time, 3),
        "generation_debug": gen_debug
    }
