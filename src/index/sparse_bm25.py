# src/index/sparse_bm25.py
import os
import json
import pickle
import re
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi

BM25_DIR = "indexes"
BM25_PATH = os.path.join(BM25_DIR, "bm25.pkl")
META_PATH = os.path.join(BM25_DIR, "bm25_meta.pkl")

_word_re = re.compile(r"[A-Za-z0-9]+")

def tokenize(text: str) -> List[str]:
    # simple + fast tokenization (better than split)
    return _word_re.findall((text or "").lower())

def build_bm25_index(corpus_path: str = "data/corpus_chunks.jsonl") -> Dict:
    """
    Build BM25 index over chunk texts and persist to disk.
    Saves:
      - bm25.pkl
      - bm25_meta.pkl (aligned with docs order)
    """
    os.makedirs(BM25_DIR, exist_ok=True)

    tokenized_docs = []
    metas = []

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            tokens = tokenize(row["text"])
            if not tokens:
                continue

            tokenized_docs.append(tokens)
            metas.append({
                "chunk_id": row["chunk_id"],
                "url": row["url"],
                "title": row["title"],
                "text": row["text"],
                "chunk_index": row.get("chunk_index", None),
            })

    bm25 = BM25Okapi(tokenized_docs)

    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)
    with open(META_PATH, "wb") as f:
        pickle.dump(metas, f)

    return {
        "bm25_docs_indexed": len(tokenized_docs),
        "bm25_path": BM25_PATH,
        "meta_path": META_PATH
    }

def sparse_retrieve(query: str, top_k: int = 20) -> List[Dict]:
    """
    Retrieve top-K chunks using BM25.
    Returns list of dicts with ranks + scores.
    """
    if not os.path.exists(BM25_PATH) or not os.path.exists(META_PATH):
        raise FileNotFoundError("BM25 index not found. Run build_bm25_index() first.")

    with open(BM25_PATH, "rb") as f:
        bm25 = pickle.load(f)
    with open(META_PATH, "rb") as f:
        metas = pickle.load(f)

    q_tokens = tokenize(query)
    if not q_tokens:
        return []

    scores = bm25.get_scores(q_tokens)  # score per doc index

    # get top_k indices by score (descending)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

    out = []
    for rank, (idx, score) in enumerate(ranked, start=1):
        m = metas[idx]
        out.append({
            "chunk_id": m["chunk_id"],
            "url": m["url"],
            "title": m["title"],
            "text": m["text"],
            "bm25_rank": rank,
            "bm25_score": float(score),
        })
    return out
