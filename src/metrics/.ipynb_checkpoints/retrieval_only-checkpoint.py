from src.index.dense_chroma import dense_retrieve
from src.index.sparse_bm25 import sparse_retrieve
from src.rag.rrf import rrf_fusion

def retrieval_only(query: str, top_k_dense=50, top_k_sparse=50, top_n_context=50, rrf_k=60):
    dense = dense_retrieve(query, top_k=top_k_dense)
    sparse = sparse_retrieve(query, top_k=top_k_sparse)
    fused = rrf_fusion(dense, sparse, k=rrf_k, top_n=top_n_context)
    return fused
