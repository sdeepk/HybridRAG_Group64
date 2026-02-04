# error_analysis_simple.py
# ------------------------------------------------------------
# Run:
#   python error_analysis_simple.py
# ------------------------------------------------------------

import os
import json
import re
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt

# ======= Update these imports only if your paths differ =======
from src.index.dense_chroma import dense_retrieve
from src.index.sparse_bm25 import sparse_retrieve
from src.rag.rrf import rrf_fusion
from src.rag.generate_flan_t5 import generate_answer_flan_t5
# ============================================================

QUESTIONS_PATH = "data/questions_100.jsonl"
OUT_DIR = "data/eval"

TOP_K_DENSE = 50
TOP_K_SPARSE = 50
TOP_N_CONTEXT = 50
RRF_K = 60

LLM_CTX_CHUNKS = 10
MAX_INPUT_TOKENS = 1536
MAX_NEW_TOKENS = 200

FAITH_THRESHOLD = 0.30  # for context_failure


def retrieval_only(query: str, top_k_dense=50, top_k_sparse=50, top_n_context=50, rrf_k=60):
    dense = dense_retrieve(query, top_k=top_k_dense)
    sparse = sparse_retrieve(query, top_k=top_k_sparse)
    fused = rrf_fusion(dense, sparse, k=rrf_k, top_n=top_n_context)
    return fused


def load_questions_jsonl(path: str):
    qs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            qs.append(json.loads(line))
    return qs


def unique_url_ranking(fused_chunks):
    urls = []
    for c in fused_chunks:
        u = c.get("url")
        if u and u not in urls:
            urls.append(u)
    return urls


def reciprocal_rank_url_level(fused_chunks, gt_urls):
    gt = set(gt_urls or [])
    ranked_urls = unique_url_ranking(fused_chunks)
    for rank, u in enumerate(ranked_urls, start=1):
        if u in gt:
            return 1.0 / rank, rank
    return 0.0, None


def split_sentences(text: str):
    return [s.strip() for s in re.split(r"[.?!]\s+", (text or "").replace("\n", " ")) if s.strip()]


def faithfulness_overlap(answer: str, fused_chunks, n_ctx=10, min_overlap=0.2, max_ctx_chars=5000):
    """
    Simple faithfulness: sentence-level token overlap with context.
    Returns 0..1 (higher = more grounded).
    """
    if not answer or not answer.strip():
        return 0.0

    context = " ".join([c.get("text", "") for c in fused_chunks[:n_ctx]]).lower()
    context = context[:max_ctx_chars]

    sents = split_sentences(answer.lower())
    if not sents:
        return 0.0

    supported = 0
    for s in sents:
        tokens = [t for t in re.findall(r"[a-zA-Z0-9]+", s) if len(t) > 3]
        if not tokens:
            continue
        overlap = sum(1 for t in tokens if t in context) / len(tokens)
        if overlap >= min_overlap:
            supported += 1

    return supported / len(sents)


def classify_error(row, faith_threshold=0.30):
    cat = str(row.get("category", "")).lower().strip()

    # Unanswerable hallucination check
    if cat == "unanswerable":
        ans = str(row.get("generated_answer", "")).strip().lower()
        if ans != "not available in context":
            return "generation_failure"
        return "no_error"

    # Retrieval failure: correct URL never retrieved
    if float(row.get("reciprocal_rank", 0.0) or 0.0) == 0.0:
        return "retrieval_failure"

    # Context failure: source retrieved but grounding weak
    if float(row.get("faithfulness", 0.0) or 0.0) < faith_threshold:
        return "context_failure"

    return "no_error"


def plot_errors_by_category(error_summary_df, out_path):
    if error_summary_df.empty:
        return
    pivot = error_summary_df.pivot(index="category", columns="error_type", values="count").fillna(0)
    pivot.plot(kind="bar", stacked=True, figsize=(8, 5))
    plt.title("Error Analysis by Question Type")
    plt.xlabel("Question Category")
    plt.ylabel("Number of Failures")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_overall_error_distribution(df, out_path):
    err = df[df["error_type"] != "no_error"]["error_type"].value_counts()
    plt.figure(figsize=(7, 4))
    err.plot(kind="bar")
    plt.title("Overall Error Type Distribution")
    plt.xlabel("Error Type")
    plt.ylabel("Number of Questions")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def error_analysis():
    os.makedirs(OUT_DIR, exist_ok=True)

    questions = load_questions_jsonl(QUESTIONS_PATH)
    rows = []

    for q in questions:
        qid = q.get("qid", "")
        query = q["question"]
        category = q.get("category", "factual")
        gt_urls = q.get("source_urls", [])

        fused = retrieval_only(
            query=query,
            top_k_dense=TOP_K_DENSE,
            top_k_sparse=TOP_K_SPARSE,
            top_n_context=TOP_N_CONTEXT,
            rrf_k=RRF_K
        )

        rr, first_rank = reciprocal_rank_url_level(fused, gt_urls)

        contexts = [c.get("text", "") for c in fused[:LLM_CTX_CHUNKS]]
        answer, _debug = generate_answer_flan_t5(
            query=query,
            contexts=contexts,
            max_input_tokens=MAX_INPUT_TOKENS,
            max_new_tokens=MAX_NEW_TOKENS
        )

        faith = faithfulness_overlap(answer, fused, n_ctx=LLM_CTX_CHUNKS)

        rows.append({
            "qid": qid,
            "question": query,
            "category": category,
            "gt_urls": gt_urls,
            "first_correct_url_rank": first_rank,
            "reciprocal_rank": rr,
            "faithfulness": faith,
            "generated_answer": answer,
        })

    # Create df
    df = pd.DataFrame(rows)

    # Classify errors
    df["error_type"] = df.apply(lambda r: classify_error(r, faith_threshold=FAITH_THRESHOLD), axis=1)

    # Save df
    df_path = os.path.join(OUT_DIR, "df_error_analysis.csv")
    df.to_csv(df_path, index=False)

    # Error summary by category
    err_df = df[df["error_type"] != "no_error"].copy()
    error_summary = (
        err_df.groupby(["category", "error_type"])
              .size()
              .reset_index(name="count")
    )
    error_summary_path = os.path.join(OUT_DIR, "error_summary.csv")
    error_summary.to_csv(error_summary_path, index=False)

    # Plots
    plot_overall_error_distribution(df, os.path.join(OUT_DIR, "overall_error_distribution.png"))
    plot_errors_by_category(error_summary, os.path.join(OUT_DIR, "errors_by_category_stacked.png"))

    print("Done")
    print("Saved df:", df_path)
    print("Saved error summary:", error_summary_path)
    print("Saved plots in:", OUT_DIR)
    print("\nError counts:")
    print(df["error_type"].value_counts())


if __name__ == "__main__":
    main()
