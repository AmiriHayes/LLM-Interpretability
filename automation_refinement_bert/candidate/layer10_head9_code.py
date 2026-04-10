import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase


def token_pair_distance_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    words = sentence.replace('[CLS]', '').replace('[SEP]', '').split()
    # Assign token positions only for valid word parts
    token_positions = {tok_id: i for i, tok_id in enumerate(toks.input_ids[0]) if tok_id < len(words)}

    for i in range(len_seq):
        for j in range(len_seq):
            # Add attentions based on token distance
            if i != j:
                out[i, j] = 1 / (1 + abs(token_positions.get(i, 0) - token_positions.get(j, 0)))

    # Normalize attention matrix
    out /= out.sum(axis=1, keepdims=True)
    return "Token Pair Distance Attention", out