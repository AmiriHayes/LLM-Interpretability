from transformers import PreTrainedTokenizerBase
import numpy as np


def sentence_boundary_emphasis(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Set strong attention to beginning of the sentence except the first token (cls)
    for token_idx in range(1, len_seq-1):
        out[token_idx, 0] = 0.83  # Attention focusing heavily on <s>

    # Set strong attention to end of the sentence marker
    out[-1, 0] = 1.0  # <s> gets the most attention

    cls_idx, eos_idx = 0, len_seq - 1
    out[0, cls_idx] = 1.0  # Self-attention for <s>
    out[eos_idx, eos_idx] = 1.0  # Self-attention for </s>

    # Ensure each row sums to 1 by setting minimum attention on the last token
    out += 1e-4  # Small amount to prevent division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize

    return "Sentence Boundary Emphasis", out