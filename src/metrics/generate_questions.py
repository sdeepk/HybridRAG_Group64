import json, random, re
from collections import defaultdict
from pathlib import Path

def load_chunks(jsonl_path: str):
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

YEAR_RE = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")
NUM_RE = re.compile(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\b\d+(?:\.\d+)?\b")

def pick_intro_text(chunks_for_url, max_chars=1200):
    # take first 1-2 chunks (assuming corpus stored in doc order)
    text = " ".join(c["text"] for c in chunks_for_url[:2])
    return text[:max_chars]

def make_factual_qas(url, title, intro_text, chunk_ids):
    qas = []
    # 1) definition-ish: "X is ..." → "What is X?"
    if " is " in intro_text[:300].lower():
        qas.append({
            "question": f"What is {title}?",
            "answer": intro_text.split(".")[0].strip() + "."
        })

    # 2) year-based: pick a year and ask "In which year...?"
    years = YEAR_RE.findall(intro_text)
    if years:
        y = years[0]
        qas.append({
            "question": f"In which year is {title} mentioned as being founded/established/formed (as per the context)?",
            "answer": y
        })

    # 3) numeric: population/area/etc. (generic)
    nums = NUM_RE.findall(intro_text)
    if nums:
        n = nums[0]
        qas.append({
            "question": f"Give one important numeric figure mentioned for {title} in the context.",
            "answer": str(n)
        })

    # attach metadata
    out = []
    for qa in qas:
        out.append({
            **qa,
            "category": "factual",
            "source_urls": [url],
            "supporting_chunk_ids": chunk_ids[:2]
        })
    return out

def make_inferential_qas(url, title, intro_text, chunk_ids):
    qas = []
    # Look for causal connectors
    lowers = intro_text.lower()
    for kw in ["because", "due to", "therefore", "as a result"]:
        if kw in lowers:
            # take sentence containing kw
            sent = next((s.strip() for s in intro_text.split(".") if kw in s.lower()), None)
            if sent:
                qas.append({
                    "question": f"Why (as per the context) is {title} described in this way?",
                    "answer": sent.strip() + ".",
                    "category": "inferential",
                    "source_urls": [url],
                    "supporting_chunk_ids": chunk_ids[:2]
                })
            break
    return qas

def make_multi_hop_qas(docA, docB):
    # simple multi-hop: ask connection via shared keyword in titles or text
    urlA, titleA, textA, chunksA = docA
    urlB, titleB, textB, chunksB = docB

    shared = None
    for token in set(re.findall(r"[A-Za-z]{4,}", titleA)) & set(re.findall(r"[A-Za-z]{4,}", titleB)):
        shared = token
        break

    # fallback: shared frequent word
    if not shared:
        wordsA = set(re.findall(r"[A-Za-z]{5,}", textA.lower()))
        wordsB = set(re.findall(r"[A-Za-z]{5,}", textB.lower()))
        common = list(wordsA & wordsB)
        if common:
            shared = common[0]

    if not shared:
        return None

    question = f"Multi-hop: What common term connects {titleA} and {titleB} in the given corpus context?"
    answer = shared
    return {
        "question": question,
        "answer": answer,
        "category": "multi-hop",
        "source_urls": [urlA, urlB],
        "supporting_chunk_ids": chunksA[:1] + chunksB[:1]
    }

def generate_questions(corpus_jsonl: str, out_jsonl: str, seed: int = 7):
    random.seed(seed)
    rows = load_chunks(corpus_jsonl)

    by_url = defaultdict(list)
    for r in rows:
        by_url[r["url"]].append(r)

    # Ensure stable order per url if chunk_id encodes order; else keep as loaded
    urls = list(by_url.keys())
    random.shuffle(urls)

    questions = []
    # Target distribution
    target = {"factual": 55, "inferential": 20, "comparative": 15, "multi-hop": 10}

    docs_for_multi = []
    numeric_docs = []

    for url in urls:
        chunks = by_url[url]
        title = chunks[0].get("title", "this page")
        intro = pick_intro_text(chunks)
        chunk_ids = [c["chunk_id"] for c in chunks]

        # factual
        if target["factual"] > 0:
            for qa in make_factual_qas(url, title, intro, chunk_ids):
                if target["factual"] <= 0: break
                questions.append(qa)
                target["factual"] -= 1

        # inferential
        if target["inferential"] > 0:
            for qa in make_inferential_qas(url, title, intro, chunk_ids):
                if target["inferential"] <= 0: break
                questions.append(qa)
                target["inferential"] -= 1

        # store docs for later
        docs_for_multi.append((url, title, intro, chunk_ids))
        if NUM_RE.search(intro):
            numeric_docs.append((url, title, intro, chunk_ids))

        if sum(target.values()) <= 0:
            break

    # comparative: pick two numeric docs, ask “which has a larger number mentioned?”
    while target["comparative"] > 0 and len(numeric_docs) >= 2:
        a, b = random.sample(numeric_docs, 2)
        numsA = [n for n in NUM_RE.findall(a[2])]
        numsB = [n for n in NUM_RE.findall(b[2])]
        if not numsA or not numsB:
            continue
        # pick first numeric mention as ground-truth comparison anchor
        qa = {
            "question": f"Comparative: Which page mentions the larger first numeric value in its intro context: {a[1]} or {b[1]}?",
            "answer": a[1] if float(numsA[0].replace(",", "")) >= float(numsB[0].replace(",", "")) else b[1],
            "category": "comparative",
            "source_urls": [a[0], b[0]],
            "supporting_chunk_ids": a[3][:1] + b[3][:1]
        }
        questions.append(qa)
        target["comparative"] -= 1

    # multi-hop
    tries = 0
    while target["multi-hop"] > 0 and tries < 200:
        tries += 1
        a, b = random.sample(docs_for_multi, 2)
        qa = make_multi_hop_qas(a, b)
        if qa:
            questions.append(qa)
            target["multi-hop"] -= 1

    # Trim or pad (should hit 100)
    questions = questions[:100]

    # Add qids
    out = []
    for i, q in enumerate(questions):
        out.append({"qid": f"q{i:04d}", **q})

    Path(out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for q in out:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    return {"count": len(out), "out": out_jsonl}

if __name__ == "__main__":
    res = generate_questions(
        corpus_jsonl="data/corpus_chunks.jsonl",
        out_jsonl="data/questions_100.jsonl",
        seed=7
    )
    print(res)
