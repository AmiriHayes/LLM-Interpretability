import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def code_structure_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenization map to align tokens
    token_map = tokenizer.convert_ids_to_tokens(toks.input_ids[0])

    # Assumed function: mainly attends to tokens that relate to code structure
    keywords = {"def", "return", "for", "if", "while", "import", "=="}

    # Assign attention based on the presence of structure keywords
    for i in range(1, len_seq - 1):
        for j in range(1, len_seq - 1):
            if token_map[i] in keywords and token_map[j] in keywords:
                out[i, j] = 1

    # Assign self-attention to [CLS] and [SEP] tokens
    out[0, 0] = 1
    out[-1, 0] = 1

    # Normalize out matrix by row to simulate attention pattern
    out += 1e-9  # Add small constant to avoid division by zero
    row_sums = out.sum(axis=1, keepdims=True)
    out = out / row_sums

    return "Code Structure Attention", out