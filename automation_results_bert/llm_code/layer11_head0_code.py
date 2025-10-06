import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def coordination_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Token indices of special tokens
    cls_token_index = toks.input_ids[0].tolist().index(tokenizer.cls_token_id)
    sep_token_index = toks.input_ids[0].tolist().index(tokenizer.sep_token_id)

    # Mark special tokens
    out[cls_token_index, cls_token_index] = 1  # CLS token self-attention
    out[sep_token_index, sep_token_index] = 1  # SEP token self-attention

    tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])

    # Find coordinating conjunctions and distribute weight to co-occurrences
    coord_indices = [i for i, tok in enumerate(tokens) if tok.lower() in ["and", "or", "but"]]

    for coord_index in coord_indices:
        # Simple pattern: Attend from conjunction to left and right neighbors
        if coord_index > 0:
            out[coord_index, coord_index - 1] = 0.5  # Attend to left token
        if coord_index < len_seq - 1:
            out[coord_index, coord_index + 1] = 0.5  # Attend to right token

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, sep_token_index] = 1.0

    return "Coordinating Conjunction and Co-occurrence", out