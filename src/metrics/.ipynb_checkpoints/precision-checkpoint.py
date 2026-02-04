def precision_at_k_url_level_one_question(fused_chunks, gt_urls, k=5):
    ranked_urls = []
    for c in fused_chunks:
        u = c["url"]
        if u not in ranked_urls:
            ranked_urls.append(u)

    topk = ranked_urls[:k]
    hits = sum(1 for u in topk if u in set(gt_urls))
    return hits / k
