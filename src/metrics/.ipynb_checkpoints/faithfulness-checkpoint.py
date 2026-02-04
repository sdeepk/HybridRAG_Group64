import re

def _sentences(text: str):
    # simple split; enough for eval
    sents = [s.strip() for s in re.split(r"[.?!]\s+", text) if s.strip()]
    return sents

def faithfulness_overlap(answer: str, fused_chunks, min_overlap=0.2):
    """
    Simple faithfulness baseline (0..1):
    Each sentence must have lexical overlap with retrieved context.
    Score = fraction of sentences supported.
    """
    context = " ".join([c["text"] for c in fused_chunks]).lower()

    sents = _sentences(answer.lower())
    if not sents:
        return 0.0

    supported = 0
    for s in sents:
        # measure token overlap ratio
        tokens = [t for t in re.findall(r"[a-zA-Z0-9]+", s) if len(t) > 3]
        if not tokens:
            continue
        overlap = sum(1 for t in tokens if t in context) / len(tokens)
        if overlap >= min_overlap:
            supported += 1

    return supported / len(sents)
