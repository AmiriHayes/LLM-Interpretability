from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase
import torch


def subject_attendance_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # Convert token IDs to tokens
    tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    # Assume first noun or pronoun is subject (simplistic heuristic)
    first_pronoun = None
    for i, token in enumerate(tokens):
        if token == 'I' or (len(token) > 1 and token[0] == '\u0120'):  # for simplicity consider stable spaces '\u0120' as start of new words (used by GPT-like models)
            first_pronoun = i
            break
    if first_pronoun is not None:
        # Highlight subject with higher attention importance
        for j in range(len_seq):
            out[first_pronoun, j] = np.exp(-abs(first_pronoun - j))  # Exponential decay from subject
    else:
        # If no prototypical subject, assign uniform attention
        out[:, :] = 1 / len_seq

    # Ensure attention across each row sums to 1
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize
    return "Subject Attention Pattern", out