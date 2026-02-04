# src/build/build_corpus.py
import os, json
from typing import List, Dict
from collections import defaultdict

from transformers import AutoTokenizer

from src.config import (
    DATA_DIR, FIXED_URLS_PATH, RANDOM_URLS_PATH, WIKI_API, MIN_WORDS
)
from src.utils.wiki_api import fetch_extract_and_props
from src.utils.validate import validate_article
from src.utils.text_clean import clean_wiki_text
from src.utils.chunking import chunk_tokens, make_chunk_id

CORPUS_PATH = f"{DATA_DIR}/corpus_chunks.jsonl"

def url_to_title(url: str) -> str:
    # https://en.wikipedia.org/wiki/Albert_Einstein -> Albert Einstein
    title = url.split("/wiki/")[-1]
    title = title.replace("_", " ")
    return title

def load_urls() -> List[str]:
    with open(FIXED_URLS_PATH, "r", encoding="utf-8") as f:
        fixed = json.load(f)
    with open(RANDOM_URLS_PATH, "r", encoding="utf-8") as f:
        rnd = json.load(f)
    urls = fixed + rnd
    # de-dupe while preserving order
    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            out.append(u)
            seen.add(u)
    return out

def build_corpus(
    chunk_size: int = 300,         # within 200-400
    overlap: int = 50,
    min_chunk_tokens: int = 200,
    tokenizer_name: str = "google/flan-t5-base",
) -> Dict:
    os.makedirs(DATA_DIR, exist_ok=True)

    urls = load_urls()
    if len(urls) != 500:
        print(f"Warning: expected 500 URLs, got {len(urls)}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    stats = defaultdict(int)
    total_chunks = 0

    with open(CORPUS_PATH, "w", encoding="utf-8") as out_f:
        for idx, url in enumerate(urls, start=1):
            title = url_to_title(url)

            # fetch text from Wikipedia by title
            try:
                extract, props = fetch_extract_and_props(WIKI_API, title)
            except Exception:
                stats["fetch_failed"] += 1
                continue

            ok, reason = validate_article(extract, props, MIN_WORDS)
            if not ok:
                stats[f"invalid_{reason}"] += 1
                continue

            clean = clean_wiki_text(extract)
            if not clean:
                stats["clean_empty"] += 1
                continue

            chunks = chunk_tokens(
                clean,
                tokenizer=tokenizer,
                chunk_size=chunk_size,
                overlap=overlap,
                min_chunk_tokens=min_chunk_tokens
            )
            if not chunks:
                stats["no_chunks_after_chunking"] += 1
                continue

            for c_i, chunk_text in enumerate(chunks):
                chunk_id = make_chunk_id(url, c_i)
                row = {
                    "chunk_id": chunk_id,
                    "url": url,
                    "title": title,
                    "chunk_index": c_i,
                    "text": chunk_text
                }
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                total_chunks += 1

            stats["pages_ok"] += 1

            # light progress
            if idx % 25 == 0:
                print(f"Processed {idx}/{len(urls)} URLs | pages_ok={stats['pages_ok']} | chunks={total_chunks}")

    return {
        "urls_seen": len(urls),
        "pages_ok": stats["pages_ok"],
        "total_chunks": total_chunks,
        "stats": dict(stats),
        "out": CORPUS_PATH
    }

if __name__ == "__main__":
    print(build_corpus())
