import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase


def subject_centric_attention_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    words = sentence.split()
    subject_indices = []

    # Start by identifying likely subject tokens primarily by proximity
    for i, tok in enumerate(words):
        if tok.lower() in {"he", "she", "they", "it", "lily", "mom", "her"} or tok.istitle():
            subject_indices.append(i)

    # Apply attention pattern: Subjects attend highly to themselves and less strong focal words
    for subj_idx in subject_indices:
        out[subj_idx + 1, subj_idx + 1] = 1.0  # self-attention for direct subject token
        # Apply descending attention-weight based on proximity
        for offset in range(-3, 4):
            if 0 <= subj_idx + offset < len(words):
                out[subj_idx + 1, subj_idx + offset + 1] = max(0.1, 1.0 / (abs(offset) + 1))

    # Ensure no row is all zeros by focusing CLS token on anything not attended to
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the output
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return "Subject-Centric Focus", out