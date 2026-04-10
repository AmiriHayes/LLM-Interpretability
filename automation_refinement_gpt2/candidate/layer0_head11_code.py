from transformers import PreTrainedTokenizerBase
import numpy as np


def sentence_initial_token_emphasis(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # Assume sentence initial token holds a significant emphasis
    initial_token_index = 1 # Assuming CLS/BOS is at index 0
    for i in range(1, len_seq - 1):
        out[initial_token_index, i] = 1
    # Normalize the out matrix row-wise
    out = out / out.sum(axis=1, keepdims=True)
    # Ensure no row is all zeros by adding small weights to EOS token
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0
    return "Sentence-Initial Token Emphasis Pattern", out