# src/rag/rrf.py
from collections import defaultdict
from typing import List, Dict

def rrf_fusion(
    dense_results: List[Dict],
    sparse_results: List[Dict],
    k: int = 60,
    top_n: int = 8
) -> List[Dict]:
    """
    Reciprocal Rank Fusion (RRF)

    dense_results: list with keys [chunk_id, dense_rank, ...]
    sparse_results: list with keys [chunk_id, bm25_rank, ...]
    """

    rrf_scores = defaultdict(float)
    chunk_store = {}

    # Dense contribution
    for d in dense_results:
        cid = d["chunk_id"]
        rank = d["dense_rank"]
        rrf_scores[cid] += 1.0 / (k + rank)
        chunk_store[cid] = d  # keep metadata

    # Sparse contribution
    for s in sparse_results:
        cid = s["chunk_id"]
        rank = s["bm25_rank"]
        rrf_scores[cid] += 1.0 / (k + rank)
        if cid not in chunk_store:
            chunk_store[cid] = s

    # Sort by RRF score (descending)
    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    final_chunks = []
    for cid, score in ranked[:top_n]:
        item = chunk_store[cid]
        item["rrf_score"] = score
        final_chunks.append(item)

    return final_chunks
