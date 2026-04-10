import numpy as np
from transformers import PreTrainedTokenizerBase

def sentence_boundary_attention(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Attention to special tokens <s> (start) and </s> (end)
    out[0, 0] = 1  # <s> attends to itself
    out[-1, -1] = 1  # </s> attends to itself

    # All tokens attend to sentence start <s> and sentence end </s>
    for idx in range(1, len_seq - 1):
        out[idx, 0] = 1  # attention to <s>
        out[idx, -1] = 1  # attention to </s>

    # Normalize the attention matrix
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)
    return "Sentence Boundary Attention", out