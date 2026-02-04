import time
import requests
from typing import List, Dict, Tuple

DEFAULT_HEADERS = {
    # Wikipedia requests a descriptive User-Agent
    "User-Agent": "HybridRAGAssignment/1.0 (contact: your_email@example.com)",
    "Accept": "application/json",
}

def _get_json(api: str, params: dict, retries: int = 5, backoff: float = 1.5) -> dict:
    last_err = None
    for i in range(retries):
        try:
            r = requests.get(api, params=params, headers=DEFAULT_HEADERS, timeout=30)

            # If Wikipedia throttles
            if r.status_code == 429:
                time.sleep(backoff ** (i + 1))
                continue

            r.raise_for_status()

            # Sometimes it returns HTML; guard before .json()
            ct = (r.headers.get("Content-Type") or "").lower()
            if "application/json" not in ct and "json" not in ct:
                # Print a short debug snippet
                snippet = (r.text or "")[:200]
                raise ValueError(f"Non-JSON response (content-type={ct}). Snippet: {snippet}")

            return r.json()

        except Exception as e:
            last_err = e
            time.sleep(backoff ** (i + 1))

    raise RuntimeError(f"Failed to fetch JSON after {retries} retries. Last error: {last_err}")

def category_members(api: str, category: str, limit: int = 200) -> List[str]:
    titles = []
    cmcontinue = None

    while len(titles) < limit:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmnamespace": 0,
            "cmlimit": min(50, limit - len(titles)),
            "format": "json",
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue

        data = _get_json(api, params)
        members = data.get("query", {}).get("categorymembers", [])
        titles.extend([m["title"] for m in members])

        cmcontinue = data.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            break

    print("Category ", category)
    print("Titles ", titles)
    print()
    return titles

def random_titles(api: str, limit: int = 50) -> List[str]:
    params = {
        "action": "query",
        "list": "random",
        "rnnamespace": 0,
        "rnlimit": limit,
        "format": "json",
    }
    data = _get_json(api, params)
    return [x["title"] for x in data.get("query", {}).get("random", [])]

def fetch_extract_and_props(api: str, title: str) -> Tuple[str, Dict]:
    params = {
        "action": "query",
        "prop": "extracts|pageprops",
        "explaintext": 1,
        "titles": title,
        "format": "json",
    }
    data = _get_json(api, params)
    pages = data.get("query", {}).get("pages", {})
    page = next(iter(pages.values()), {})
    extract = page.get("extract", "") or ""
    props = page.get("pageprops", {}) or {}
    return extract, props

def title_to_url(title: str) -> str:
    return "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")
