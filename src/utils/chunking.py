# src/utils/chunking.py
from typing import List, Dict, Any
import hashlib

def make_chunk_id(url: str, chunk_index: int) -> str:
    # Stable ID: short hash(url) + chunk_index
    h = hashlib.md5(url.encode("utf-8")).hexdigest()[:10]
    return f"{h}_c{chunk_index:04d}"

def chunk_tokens(
    text: str,
    tokenizer,
    chunk_size: int = 300,        # within 200-400
    overlap: int = 50,
    min_chunk_tokens: int = 200,  # enforce requirement lower bound
) -> List[str]:
    """
    Token-based chunking with overlap.
    - Produces chunks of ~chunk_size tokens
    - Overlap between consecutive chunks
    - If last chunk < min_chunk_tokens, merge into previous (if possible)
    """
    if not text or not text.strip():
        return []

    # Encode to token IDs (no special tokens)
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    if len(token_ids) <= chunk_size:
        # single chunk; ensure min tokens
        return [text] if len(token_ids) >= min_chunk_tokens else []

    chunks_ids = []
    step = max(1, chunk_size - overlap)

    for start in range(0, len(token_ids), step):
        end = start + chunk_size
        piece = token_ids[start:end]
        if not piece:
            break
        chunks_ids.append(piece)
        if end >= len(token_ids):
            break

    # Handle tail chunk too small: merge with previous
    if len(chunks_ids) >= 2 and len(chunks_ids[-1]) < min_chunk_tokens:
        chunks_ids[-2] = chunks_ids[-2] + chunks_ids[-1]
        chunks_ids.pop()

    # Decode back to text
    chunks_text = [tokenizer.decode(ids, skip_special_tokens=True) for ids in chunks_ids]

    # Final guard: remove any chunk still < min_chunk_tokens
    out = []
    for ch in chunks_text:
        ids = tokenizer.encode(ch, add_special_tokens=False)
        if len(ids) >= min_chunk_tokens:
            out.append(ch.strip())
    return out
