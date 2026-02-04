# src/utils/text_clean.py
import re

def clean_wiki_text(text: str) -> str:
    """
    MediaWiki extracts are already plain text, but we still:
    - normalize whitespace
    - remove very common boilerplate headings
    """
    if not text:
        return ""
    t = re.sub(r"\s+", " ", text).strip()

    # Optional: remove common end sections to reduce noise
    # (safe heuristic; not mandatory)
    for marker in ["See also", "References", "External links", "Further reading"]:
        # cut if marker appears as a heading-like pattern
        idx = t.lower().find(marker.lower())
        if idx != -1 and idx > len(t) * 0.5:
            t = t[:idx].strip()
    return t
