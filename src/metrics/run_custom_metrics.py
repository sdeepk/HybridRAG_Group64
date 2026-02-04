from src.index.dense_chroma import dense_retrieve
from src.index.sparse_bm25 import sparse_retrieve
from src.rag.rrf import rrf_fusion

import json
import time

from src.rag.generate_flan_t5 import generate_answer_flan_t5

from src.metrics.mrr import mrr_url_level_one_question
from src.metrics.precision import precision_at_k_url_level_one_question
from src.metrics.faithfulness import faithfulness_overlap

def retrieval_only(query: str, top_k_dense=50, top_k_sparse=50, top_n_context=50, rrf_k=60):
    dense = dense_retrieve(query, top_k=top_k_dense)
    sparse = sparse_retrieve(query, top_k=top_k_sparse)
    fused = rrf_fusion(dense, sparse, k=rrf_k, top_n=top_n_context)
    return fused

def load_questions(path="data/questions_100.jsonl"):
    qs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            qs.append(json.loads(line))
    return qs


def rag_answer_from_fused(query, fused_chunks):
    contexts = [c["text"] for c in fused_chunks[:10]]
    answer, debug = generate_answer_flan_t5(query, contexts, max_input_tokens=1536, max_new_tokens=220)
    return answer


def run_custom_metrics():
    questions = load_questions()

    mrr_scores = []
    p_scores = []
    faith_scores = []
    rows = []

    for q in questions:
        query = q["question"]
        gt_urls = q["source_urls"]

        t0 = time.time()
        fused = retrieval_only(query, top_k_dense=50, top_k_sparse=50, top_n_context=50)
        t1 = time.time()

        # Retrieval metrics
        rr = mrr_url_level_one_question(fused, gt_urls)
        p5 = precision_at_k_url_level_one_question(fused, gt_urls, k=5)

        # Generation + faithfulness
        answer = rag_answer_from_fused(query, fused)
        faith = faithfulness_overlap(answer, fused)

        mrr_scores.append(rr)
        p_scores.append(p5)
        faith_scores.append(faith)

        rows.append({
            "qid": q["qid"],
            "question": query,
            "MRR_rr": rr,
            "Precision@5": p5,
            "Faithfulness": faith,
            "retrieval_time_sec": round(t1 - t0, 3),
        })

    print("MRR:", round(sum(mrr_scores)/len(mrr_scores), 4))
    print("Precision@5:", round(sum(p_scores)/len(p_scores), 4))
    print("Faithfulness:", round(sum(faith_scores)/len(faith_scores), 4))
