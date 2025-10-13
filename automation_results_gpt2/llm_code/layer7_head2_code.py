import numpy as np
from transformers import PreTrainedTokenizerBase

def sentence_start_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> str:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # The first non-padding token in the sequence is given complete focus
    focus_index = 1
    out[focus_index] = 1

    # Ensure no row is all zeros by adding slight attention to eos
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the attention matrix
    out = out / out.sum(axis=1, keepdims=True)

    return "Sentence Start Token Focus", out