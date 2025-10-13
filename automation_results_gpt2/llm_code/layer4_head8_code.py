import numpy as np
from transformers import PreTrainedTokenizerBase
def sentence_initial_dominance(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # Assuming CLS token attention to itself and the sentence start token 
    init_token_index = 1 
    out[0, init_token_index] = 1
    out[init_token_index] = 1  # Initial token pays highest attention to all
    # Ensure rest of tokens have at least attention to one end token
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0  # EOS token
    out += 1e-4  # Prevent division by zero during normalization
    out = out / out.sum(axis=1, keepdims=True)  # Normalize
    return "Sentence Initial Word Dominance", out