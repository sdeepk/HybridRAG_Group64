# src/rag/generate_flan_t5.py

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from typing import List, Tuple
import re

MODEL_ID = "google/flan-t5-large"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)


def trim_to_last_sentence(text):
    # Regex looks for the last occurrence of . ! or ?
    # If the text is empty or has no punctuation, it returns the original
    match = list(re.finditer(r'[.!?]', text))
    if match:
        last_punct_idx = match[-1].start()
        return text[:last_punct_idx + 1]
    return text

def generate_answer_flan_t5(
    query: str,
    contexts: List[str],
    max_input_tokens: int = 1536,
    max_new_tokens: int = 220
) -> Tuple[str, dict]:
    """
    Generate answer using Flan-T5-Large with strict grounding.
    """

    # Reduce each context to avoid truncation
    context_block = "\n\n---\n\n".join([c[:800] for c in contexts])

    prompt = (
        "Answer the question using ONLY the context below.\n"
        "Provide a complete and concise answer ending with a full stop.\n\n"
        "Be concise (4-5 sentences).\n"
        "If the answer is not present, say: Not available in context.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            min_length=12,
            num_beams=4,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            early_stopping=False,
            length_penalty=1.1
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    debug = {
        "model": MODEL_ID,
        "prompt_tokens": inputs["input_ids"].shape[1],
        "max_input_tokens": max_input_tokens,
        "max_new_tokens": max_new_tokens
    }
    text =  trim_to_last_sentence(answer)
    return  text, debug
