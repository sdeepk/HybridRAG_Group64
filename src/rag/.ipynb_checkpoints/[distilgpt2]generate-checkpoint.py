from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_ID = "distilgpt2"

tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

# distilgpt2 has no pad token by default
tok.pad_token = tok.eos_token

def generate_answer_distilgpt2(query, contexts, max_input_tokens=900, max_new_tokens=120):
    # Keep context compact to avoid truncation
    context_block = "\n\n---\n\n".join([c[:700] for c in contexts])  # shrink each chunk

    prompt = (
        "If the answer is not present in the context, reply exactly: Not available in context.\n\n"
        "Provide a complete and concise answer ending with a full stop.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )

    inputs = tok(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
        padding=True
    )

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,              # deterministic
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            eos_token_id=tok.eos_token_id
        )

    text = tok.decode(out[0], skip_special_tokens=True)

    # Extract only the answer part after "Answer:"
    if "Answer:" in text:
        text = text.split("Answer:", 1)[-1].strip()

    # Clean small artifacts
    return text.strip()
