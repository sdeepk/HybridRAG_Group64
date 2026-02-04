def mrr_url_level_one_question(fused_chunks, gt_urls):
    # chunk -> url ranking
    ranked_urls = []
    for c in fused_chunks:
        u = c["url"]
        if u not in ranked_urls:
            ranked_urls.append(u)

    # find first correct url rank
    for rank, u in enumerate(ranked_urls, start=1):
        if u in set(gt_urls):
            return 1.0 / rank
    return 0.0
