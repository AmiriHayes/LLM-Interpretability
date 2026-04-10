import numpy as np
from transformers import PreTrainedTokenizerBase

def initial_token_dominance(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # Set the first token to have dominant attention for all tokens
    for i in range(len_seq):
        out[i, 0] = 1.0 # The first token receives strong attention from all tokens
    # Normalize attention rows to ensure sum of attentions is 1
    out += 1e-4 # To avoid any division by zero issues
    out = out / out.sum(axis=1, keepdims=True)
    return "Initial Token Dominance", out