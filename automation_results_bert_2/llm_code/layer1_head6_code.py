import numpy as np
from transformers import PreTrainedTokenizerBase

def coordination_cooccurrence_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # Tokenize sentence split by spaces to compare as string
    token_strs = sentence.split()
    last_conj_index = -1
    for i, token in enumerate(token_strs):
        if token.lower() == 'and':
            last_conj_index = i
        if last_conj_index != -1:
            out[last_conj_index+1, i+1] = 1
            out[i+1, last_conj_index+1] = 1

    # Ensure no token row is all zeros
    for row in range(len_seq): 
        if out[row].sum() == 0:
            out[row, -1] = 1.0
    return 'Coordination and Co-occurrence Pattern', out