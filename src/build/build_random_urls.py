# src/build/build_random_urls.py
import os
import json
import random
import time
from collections import defaultdict
from typing import Set

from src.config import (
    WIKI_API, RANDOM_COUNT, MIN_WORDS,
    DATA_DIR, FIXED_URLS_PATH, RANDOM_URLS_PATH
)
from src.utils.wiki_api import random_titles, fetch_extract_and_props, title_to_url
from src.utils.validate import validate_article


def _load_fixed_urls() -> Set[str]:
    if not os.path.exists(FIXED_URLS_PATH):
        raise FileNotFoundError(
            f"fixed_urls.json not found at {FIXED_URLS_PATH}. "
            "Run build_fixed_urls() first."
        )
    with open(FIXED_URLS_PATH, "r", encoding="utf-8") as f:
        return set(json.load(f))


def build_random_urls(
    target_count: int = RANDOM_COUNT,
    batch_size: int = 50,
    max_loops: int = 300,          # safety: avoid infinite loops
    per_title_retries: int = 2,    # retry extract fetch per title (extra safety)
) -> dict:
    """
    Build a random set of Wikipedia URLs (namespace 0), ensuring:
    - min 200 words (MIN_WORDS)
    - not disambiguation
    - no overlap with fixed_urls.json
    - changes every run (time-based seed)
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    fixed_urls = _load_fixed_urls()
    picked = []
    picked_urls = set(fixed_urls)  # prevent overlap + duplicates

    reject_stats = defaultdict(int)
    error_stats = defaultdict(int)

    # new randomness per run
    random.seed(int(time.time()))

    loops = 0
    while len(picked) < target_count and loops < max_loops:
        loops += 1

        # 1) get random titles (wiki_api already has retries/backoff inside)
        try:
            titles = random_titles(WIKI_API, limit=batch_size)
        except Exception as e:
            error_stats["random_titles_failed"] += 1
            # small sleep to avoid hammering
            time.sleep(1.5)
            continue

        if not titles:
            error_stats["random_titles_empty"] += 1
            time.sleep(1.0)
            continue

        # 2) validate each title
        for title in titles:
            if len(picked) >= target_count:
                break

            url = title_to_url(title)
            if url in picked_urls:
                reject_stats["duplicate_or_overlap"] += 1
                continue

            # 2a) fetch extract with a couple of retries (extra safety)
            extract = ""
            props = {}
            ok_fetch = False

            for _ in range(per_title_retries + 1):
                try:
                    extract, props = fetch_extract_and_props(WIKI_API, title)
                    ok_fetch = True
                    break
                except Exception:
                    # backoff per title
                    time.sleep(0.8)

            if not ok_fetch:
                error_stats["extract_fetch_failed"] += 1
                continue

            # 2b) apply rules: min words + non-disambiguation + non-empty
            ok, reason = validate_article(extract, props, MIN_WORDS)
            if not ok:
                reject_stats[reason] += 1
                continue

            picked.append(url)
            picked_urls.add(url)

        # polite pacing (prevents rate limiting)
        if loops % 5 == 0:
            time.sleep(0.5)

    if len(picked) < target_count:
        raise RuntimeError(
            f"Could not collect {target_count} random URLs. Collected {len(picked)} "
            f"after {loops} loops. Errors={dict(error_stats)} Rejects={dict(reject_stats)}"
        )

    with open(RANDOM_URLS_PATH, "w", encoding="utf-8") as f:
        json.dump(picked, f, indent=2, ensure_ascii=False)

    return {
        "random_count": len(picked),
        "loops": loops,
        "rejections": dict(reject_stats),
        "errors": dict(error_stats),
        "out": RANDOM_URLS_PATH
    }


if __name__ == "__main__":
    print(build_random_urls())
