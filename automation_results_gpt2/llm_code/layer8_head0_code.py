import numpy as np
from transformers import PreTrainedTokenizerBase

def initial_token_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Initial token gets the majority of the attention in each sentence
    out[0, 0] = 1.0  # Attention to itself (if the first token)
    for i in range(1, len_seq - 1):
        out[1, i] = 0.1 + 0.1 * (len_seq - i) / len_seq

    # Ensure no row is all zeros by attending to the <eos> token
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Focus on Sentence Initial Tokens", out