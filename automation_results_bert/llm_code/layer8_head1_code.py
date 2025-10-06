import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def boundary_and_quotation_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Boundary attention pattern
    out[0, -1] = 1.0  # [CLS] focuses on [SEP] and vice versa
    out[-1, 0] = 1.0

    # Identify quotation marks and set self-attention on them
    tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    quote_indices = [i for i, token in enumerate(tokens) if token in ['"', '\"']]
    for idx in quote_indices:
        out[idx, idx] = 1.0  # Attention to self for quotes

    # Falling back on ensuring no row is purely zero
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalizing each row to ensure the sum of attention weights equals 1
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return "Sentence Boundary and Quotation Attention", out
