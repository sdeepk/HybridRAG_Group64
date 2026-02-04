from src.index.dense_chroma import dense_retrieve
from src.index.sparse_bm25 import sparse_retrieve
from src.rag.rrf import rrf_fusion
import json
import time
import pandas as pd
from itertools import product

from src.metrics.mrr import mrr_url_level_one_question
from src.metrics.precision import precision_at_k_url_level_one_question
from src.metrics.faithfulness import faithfulness_overlap

from src.rag.generate_flan_t5 import generate_answer_flan_t5

def retrieve_mode(query: str,
                  mode: str,
                  top_k_dense: int = 50,
                  top_k_sparse: int = 50,
                  top_n_context: int = 50,
                  rrf_k: int = 60):
    """
    Returns fused_chunks-like list of dicts with at least: chunk_id, url, text
    mode: "dense" | "sparse" | "hybrid"
    """
    mode = mode.lower()

    if mode == "dense":
        return dense_retrieve(query, top_k=top_k_dense)[:top_n_context]

    if mode == "sparse":
        return sparse_retrieve(query, top_k=top_k_sparse)[:top_n_context]

    if mode == "hybrid":
        dense = dense_retrieve(query, top_k=top_k_dense)
        sparse = sparse_retrieve(query, top_k=top_k_sparse)
        return rrf_fusion(dense, sparse, k=rrf_k, top_n=top_n_context)

    raise ValueError(f"Unknown mode: {mode}")

def load_questions(path):
    qs=[]
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            qs.append(json.loads(line))
    return qs

def generate_answer_from_chunks(query, fused_chunks, n_ctx=10):
    contexts = [c["text"] for c in fused_chunks[:n_ctx]]
    ans, _dbg = generate_answer_flan_t5(query, contexts, max_input_tokens=1536, max_new_tokens=220)
    return ans

def eval_config(questions, mode, kd, ks, n, rrf_k, precision_k=5, with_generation=False):
    rr_list=[]
    p_list=[]
    faith_list=[]
    retr_times=[]
    total_times=[]

    for q in questions:
        query = q["question"]
        gt_urls = q["source_urls"]

        t0=time.time()
        fused = retrieve_mode(
            query=query,
            mode=mode,
            top_k_dense=kd,
            top_k_sparse=ks,
            top_n_context=max(n, 50),   # retrieval list size for metrics
            rrf_k=rrf_k
        )
        t1=time.time()

        rr = mrr_url_level_one_question(fused, gt_urls)
        p  = precision_at_k_url_level_one_question(fused, gt_urls, k=precision_k)

        rr_list.append(rr)
        p_list.append(p)
        retr_times.append(t1-t0)

        if with_generation:
            ans = generate_answer_from_chunks(query, fused, n_ctx=n)
            faith = faithfulness_overlap(ans, fused[:n])
            faith_list.append(faith)
            total_times.append(time.time()-t0)

    out = {
        "mode": mode,
        "top_k_dense": kd,
        "top_k_sparse": ks,
        "N_context": n,
        "rrf_k": (rrf_k if mode=="hybrid" else None),
        "MRR": sum(rr_list)/len(rr_list),
        f"Precision@{precision_k}": sum(p_list)/len(p_list),
        "avg_retrieval_time": sum(retr_times)/len(retr_times)
    }
    if with_generation:
        out["Faithfulness"] = sum(faith_list)/len(faith_list) if faith_list else None
        out["avg_total_time"] = sum(total_times)/len(total_times) if total_times else None

    return out

def run_ablation():
    questions = load_questions("data/questions_100.jsonl")

    modes = ["dense", "sparse", "hybrid"]
    K_vals = [10, 30, 50]
    N_vals = [5, 10]
    RRF_vals = [10, 60]   # used only for hybrid

    rows = []

    for mode in modes:
        print("Mode-",mode)
        for K in K_vals:
            print("K-", K)
            for N in N_vals:
                print("N-", N)
                if mode == "hybrid":
                    for rrfk in RRF_vals:
                        rows.append(eval_config(questions, mode, kd=K, ks=K, n=N, rrf_k=rrfk,
                                                precision_k=5, with_generation=False))
                elif mode == "dense":
                    rows.append(eval_config(questions, mode, kd=K, ks=K, n=N, rrf_k=60,
                                            precision_k=5, with_generation=False))
                else:  # sparse
                    rows.append(eval_config(questions, mode, kd=K, ks=K, n=N, rrf_k=60,
                                            precision_k=5, with_generation=False))

    df = pd.DataFrame(rows)
    df.to_csv("data/eval_run/ablation_results.csv", index=False)

    print(df.sort_values("MRR", ascending=False).head(10))
    return df

if __name__ == "__main__":
   generate_matrix()