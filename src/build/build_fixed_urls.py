# src/build/build_fixed_urls.py
import os, json, random
from collections import defaultdict

from src.config import (
    WIKI_API, CATEGORY_SEEDS, FIXED_COUNT, MIN_WORDS,
    DATA_DIR, FIXED_URLS_PATH, SEED
)
from src.utils.wiki_api import category_members, fetch_extract_and_props, title_to_url
from src.utils.validate import validate_article

def build_fixed_urls():
    os.makedirs(DATA_DIR, exist_ok=True)
    random.seed(SEED)

    picked = []
    picked_titles = set()
    reject_stats = defaultdict(int)

    # Round-robin across categories to enforce diversity
    cat_lists = {}
    for c in CATEGORY_SEEDS:
        print(c)
        titles = category_members(WIKI_API, c, limit=400)
        random.shuffle(titles)
        cat_lists[c] = titles

    cat_idx = 0
    cats = list(CATEGORY_SEEDS)

    while len(picked) < FIXED_COUNT:
        c = cats[cat_idx % len(cats)]
        cat_idx += 1

        if not cat_lists[c]:
            continue

        title = cat_lists[c].pop()
        if title in picked_titles:
            continue

        extract, props = fetch_extract_and_props(WIKI_API, title)
        ok, reason = validate_article(extract, props, MIN_WORDS)
        if not ok:
            reject_stats[reason] += 1
            continue

        url = title_to_url(title)
        picked.append(url)
        picked_titles.add(title)

    with open(FIXED_URLS_PATH, "w", encoding="utf-8") as f:
        json.dump(picked, f, indent=2, ensure_ascii=False)

    return {
        "fixed_count": len(picked),
        "rejections": dict(reject_stats),
        "out": FIXED_URLS_PATH
    }

if __name__ == "__main__":
    print(build_fixed_urls())
