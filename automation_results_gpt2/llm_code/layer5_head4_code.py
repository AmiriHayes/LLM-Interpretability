import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def sentence_start_anchor(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # The first token (excluding special tokens) is responsible for anchoring most of the attention
    # Assuming token at index 1 is the start of the actual sentence for GPT type models
    starting_token_idx = 1
    for i in range(starting_token_idx, len_seq - 1):  # Exclude the [CLS] and [SEP] or similar
        out[starting_token_idx, i] = 1

    # Ensure no row is all zeros (for special tokens or punctuation typically handled separately)
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4  # Avoid division by zero if applicable
    out = out / out.sum(axis=1, keepdims=True)  # Normalize row attention

    return "Sentence Start Anchor", out