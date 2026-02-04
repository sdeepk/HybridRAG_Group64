# src/eval/metrics_pipeline.py
# ------------------------------------------------------------
# Run:
#   python src/eval/metrics_pipeline.py
# ------------------------------------------------------------

import os, json, time, re
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt

from src.index.dense_chroma import dense_retrieve
from src.index.sparse_bm25 import sparse_retrieve
from src.rag.rrf import rrf_fusion
from src.rag.generate_flan_t5 import generate_answer_flan_t5
from src.metrics.advance_run_ablation import run_ablation

QUESTIONS_PATH = "data/questions_100.jsonl"
OUT_DIR = "data/eval_run"

TOP_K_DENSE = 50
TOP_K_SPARSE = 50
TOP_N_CONTEXT = 50
RRF_K = 60

LLM_CTX_CHUNKS = 10
MAX_INPUT_TOKENS = 1536
MAX_NEW_TOKENS = 220

FAITH_THRESHOLD = 0.30
PRECISION_K = 5


# ==========================
# RETRIEVAL
# ==========================
def retrieval_only(query: str):
    dense = dense_retrieve(query, top_k=TOP_K_DENSE)
    sparse = sparse_retrieve(query, top_k=TOP_K_SPARSE)
    return rrf_fusion(dense, sparse, k=RRF_K, top_n=TOP_N_CONTEXT)


# ==========================
# METRICS
# ==========================
def unique_url_ranking(fused):
    urls = []
    for c in fused:
        u = c.get("url")
        if u and u not in urls:
            urls.append(u)
    return urls


def reciprocal_rank_url_level(fused, gt_urls):
    gt = set(gt_urls or [])
    for rank, u in enumerate(unique_url_ranking(fused), start=1):
        if u in gt:
            return 1.0 / rank, rank
    return 0.0, None


def precision_at_k_url_level(fused, gt_urls, k=5):
    gt = set(gt_urls or [])
    topk = unique_url_ranking(fused)[:k]
    return (sum(1 for u in topk if u in gt) / k) if k else 0.0


def split_sentences(text):
    return [s.strip() for s in re.split(r"[.?!]\s+", (text or "").replace("\n", " ")) if s.strip()]


def faithfulness_overlap(answer, fused):
    """
    Simple faithfulness: fraction of answer sentences supported by context via token overlap.
    0..1 (higher is better)
    """
    if not answer or not answer.strip():
        return 0.0

    context = " ".join(c.get("text", "") for c in fused[:LLM_CTX_CHUNKS]).lower()
    sents = split_sentences(answer.lower())
    if not sents:
        return 0.0

    supported = 0
    for s in sents:
        tokens = [t for t in re.findall(r"\w+", s) if len(t) > 3]
        if not tokens:
            continue
        overlap = sum(1 for t in tokens if t in context) / len(tokens)
        if overlap >= 0.2:
            supported += 1

    return supported / len(sents)


def classify_error(row):
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

    # Context failure: retrieved correct source but weak grounding
    if float(row.get("faithfulness", 0.0) or 0.0) < FAITH_THRESHOLD:
        return "context_failure"

    return "no_error"


# ==========================
# LOAD QUESTIONS
# ==========================
def load_questions(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# ==========================
# VISUALIZATIONS
# ==========================
def save_rank_distribution(df, out_path, max_rank=10):
    ranks = df["first_correct_url_rank"].dropna().astype(int).tolist()
    counts = Counter(ranks)
    xs = list(range(1, max_rank + 1))
    ys = [counts.get(i, 0) for i in xs]

    plt.figure(figsize=(8, 4))
    plt.bar(xs, ys)
    plt.title("Rank Distribution: First Correct URL")
    plt.xlabel("Rank")
    plt.ylabel("Number of Questions")
    plt.xticks(xs)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_error_overall_bar(df, out_path):
    err = df[df["error_type"] != "no_error"]["error_type"].value_counts()
    plt.figure(figsize=(7, 4))
    err.plot(kind="bar")
    plt.title("Overall Error Type Distribution")
    plt.xlabel("Error Type")
    plt.ylabel("Count")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_error_by_category_stacked(error_summary, out_path):
    if error_summary.empty:
        return
    pivot = error_summary.pivot(index="category", columns="error_type", values="count").fillna(0)
    pivot.plot(kind="bar", stacked=True, figsize=(9, 5))
    plt.title("Error Analysis by Question Type (stacked)")
    plt.xlabel("Question Category")
    plt.ylabel("Failures")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_latency_hist(df, out_path):
    plt.figure(figsize=(8, 4))
    df["total_time_sec"].plot(kind="hist", bins=20)
    plt.title("Latency Distribution (total_time_sec)")
    plt.xlabel("Seconds")
    plt.ylabel("Count")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_metric_by_category(df, out_path):
    """
    Shows avg MRR + avg Precision@5 per category.
    """
    if "category" not in df.columns:
        return
    agg = df.groupby("category")[["reciprocal_rank", "precision@5", "faithfulness"]].mean().sort_values(
        "reciprocal_rank", ascending=False
    )

    plt.figure(figsize=(9, 5))
    agg["reciprocal_rank"].plot(kind="bar")
    plt.title("Average MRR (URL-level) by Question Category")
    plt.xlabel("Category")
    plt.ylabel("Avg MRR")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path.replace(".png", "_mrr.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(9, 5))
    agg["precision@5"].plot(kind="bar")
    plt.title("Average Precision@5 (URL-level) by Question Category")
    plt.xlabel("Category")
    plt.ylabel("Avg Precision@5")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path.replace(".png", "_p5.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(9, 5))
    agg["faithfulness"].plot(kind="bar")
    plt.title("Average Faithfulness by Question Category")
    plt.xlabel("Category")
    plt.ylabel("Avg Faithfulness")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path.replace(".png", "_faith.png"), dpi=200)
    plt.close()


# ==========================
# MAIN PIPELINE
# ==========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    questions = load_questions(QUESTIONS_PATH)

    rows = []
    for q in questions:
        qid = q.get("qid", "")
        query = q["question"]
        category = q.get("category", "factual")
        gt_urls = q.get("source_urls", [])

        t0 = time.time()
        fused = retrieval_only(query)
        t1 = time.time()

        rr, rank = reciprocal_rank_url_level(fused, gt_urls)
        p5 = precision_at_k_url_level(fused, gt_urls, k=PRECISION_K)

        answer, _ = generate_answer_flan_t5(
            query=query,
            contexts=[c.get("text", "") for c in fused[:LLM_CTX_CHUNKS]],
            max_input_tokens=MAX_INPUT_TOKENS,
            max_new_tokens=MAX_NEW_TOKENS
        )

        faith = faithfulness_overlap(answer, fused)
        t2 = time.time()

        rows.append({
            "qid": qid,
            "question": query,
            "category": category,
            "gt_urls": gt_urls,
            "first_correct_url_rank": rank,
            "reciprocal_rank": rr,
            "precision@5": p5,
            "faithfulness": faith,
            "generated_answer": answer,
            "retrieval_time_sec": round(t1 - t0, 4),
            "total_time_sec": round(t2 - t0, 4)
        })

    df = pd.DataFrame(rows)
    df["error_type"] = df.apply(classify_error, axis=1)

    # -------- Save structured outputs --------
    df.to_csv(os.path.join(OUT_DIR, "results.csv"), index=False)
    df.to_json(os.path.join(OUT_DIR, "results.json"), orient="records", indent=2, force_ascii=False)

    summary = {
        "num_questions": int(len(df)),
        "MRR_url_level": round(df["reciprocal_rank"].mean(), 4),
        "Precision@5_url_level": round(df["precision@5"].mean(), 4),
        "Faithfulness_avg": round(df["faithfulness"].mean(), 4),
        "Avg_retrieval_time_sec": round(df["retrieval_time_sec"].mean(), 4),
        "Avg_total_time_sec": round(df["total_time_sec"].mean(), 4),
        "Error_counts": df["error_type"].value_counts().to_dict()
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


    # -------- Advance Run Ablation --------
    run_ablation()
    
    # -------- Error summary by category --------
    err_df = df[df["error_type"] != "no_error"].copy()
    error_summary = (
        err_df.groupby(["category", "error_type"])
              .size()
              .reset_index(name="count")
    )
    error_summary.to_csv(os.path.join(OUT_DIR, "error_summary.csv"), index=False)

    # -------- Visualizations --------
    save_rank_distribution(df, os.path.join(OUT_DIR, "rank_distribution.png"), max_rank=10)
    save_error_overall_bar(df, os.path.join(OUT_DIR, "overall_error_distribution.png"))
    save_error_by_category_stacked(error_summary, os.path.join(OUT_DIR, "errors_by_category_stacked.png"))
    save_latency_hist(df, os.path.join(OUT_DIR, "latency_hist.png"))
    save_metric_by_category(df, os.path.join(OUT_DIR, "category_metrics.png"))

    print("Done. Outputs saved in:", OUT_DIR)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
