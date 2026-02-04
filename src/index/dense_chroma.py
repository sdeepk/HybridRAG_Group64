# src/index/dense_chroma.py
import os, json
from typing import Dict, List

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

CHROMA_DIR = "indexes/chroma"
COLLECTION_NAME = "wiki_chunks_dense"

def _get_client():
    os.makedirs(CHROMA_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_DIR)
    
def build_dense_index(
    corpus_path: str = "data/corpus_chunks.jsonl",
    embedding_model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64
) -> Dict:
    """
    Builds a persistent Chroma collection with precomputed embeddings.
    Stores metadata: url, title, chunk_id
    """
    embedder = SentenceTransformer(embedding_model_name)
    client = _get_client()

    # fresh rebuild: delete if exists
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # cosine distance
    )

    ids, docs, metas = [], [], []

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            ids.append(row["chunk_id"])
            docs.append(row["text"])
            metas.append({"url": row["url"], "title": row["title"]})

    # embed and add in batches (memory safe)
    for start in range(0, len(docs), batch_size):
        end = start + batch_size
        batch_docs = docs[start:end]
        batch_ids = ids[start:end]
        batch_metas = metas[start:end]

        embs = embedder.encode(batch_docs, show_progress_bar=False).tolist()
        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_metas,
            embeddings=embs
        )

    #client.persist()
    return {
        "collection": COLLECTION_NAME,
        "persist_dir": CHROMA_DIR,
        "num_chunks_indexed": len(docs),
        "embedding_model": embedding_model_name
    }

def dense_retrieve(
    query: str,
    top_k: int = 20,
    embedding_model_name: str = "all-MiniLM-L6-v2",
) -> List[Dict]:
    """
    Returns top_k chunks by cosine similarity.
    Output includes:
      - chunk_id
      - text
      - url, title
      - dense_score (cosine similarity approx)
      - dense_rank
    """
    embedder = SentenceTransformer(embedding_model_name)
    client = _get_client()
    collection = client.get_collection(COLLECTION_NAME)

    q_emb = embedder.encode([query]).tolist()

    res = collection.query(
        query_embeddings=q_emb,
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    out = []
    ids = res["ids"][0]
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]  # cosine distance (0 = identical)

    for rank, (cid, doc, meta, dist) in enumerate(zip(ids, docs, metas, dists), start=1):
        # cosine similarity approx = 1 - cosine_distance
        sim = 1.0 - float(dist)
        out.append({
            "chunk_id": cid,
            "text": doc,
            "url": meta.get("url"),
            "title": meta.get("title"),
            "dense_rank": rank,
            "dense_distance": float(dist),
            "dense_score": sim,
        })

    return out
