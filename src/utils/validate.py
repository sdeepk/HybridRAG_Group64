import re
from typing import Tuple

def clean_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()

def is_disambiguation(pageprops: dict) -> bool:
    # MediaWiki marks disambiguation pages with pageprops.disambiguation
    return pageprops.get("disambiguation") is not None

def word_count(text: str) -> int:
    t = clean_ws(text)
    return 0 if not t else len(t.split())

def validate_article(extract: str, pageprops: dict, min_words: int) -> Tuple[bool, str]:
    if not extract or not clean_ws(extract):
        return False, "empty_extract"
    if is_disambiguation(pageprops):
        return False, "disambiguation"
    wc = word_count(extract)
    if wc < min_words:
        return False, f"too_short_{wc}"
    return True, "ok"
