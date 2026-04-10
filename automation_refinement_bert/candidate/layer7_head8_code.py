import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def math_contextual_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    math_token_indices = [i for i, tok_id in enumerate(toks.input_ids[0]) if tok_id in tokenizer.encode(['+', '-', '*', '/', '=', 'x', '(', ')', '^'])]

    for idx in math_token_indices:
        for i in range(1, len_seq-1):  # Attending all to mathematical tokens
            out[i, idx] = 0.5 
            out[idx, i] = 0.5
        # Adding attention to surrounding tokens of the last math token index
        if idx != 0 and idx != len_seq - 1:
            out[idx, idx + 1] = 0.5
            out[idx, idx - 1] = 0.2

    # CLS token attends to itself
    out[0, 0] = 1
    # SEP token attends to itself
    out[len_seq - 1, len_seq - 1] = 1

    return "Math Contextual Attention Pattern", out
