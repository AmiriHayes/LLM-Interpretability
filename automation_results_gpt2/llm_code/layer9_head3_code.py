import numpy as np
from transformers import PreTrainedTokenizerBase

def first_word_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> str:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # The first token that doesn't represent special tokens likely gets attention
    first_word_idx = 1
    # Assigning high attention to the first word by letting it attend to itself
    for i in range(1, len_seq):
        out[i, first_word_idx] = 1.0
    for row in range(len_seq): # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0
    return "First Word Prediction", out