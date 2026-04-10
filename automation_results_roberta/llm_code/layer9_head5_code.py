import numpy as np
from transformers import PreTrainedTokenizerBase

def sentence_boundary_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> (str, np.ndarray):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assign high attention to sentence boundary tokens '<s>' and '</s>'
    out[0, 0] = 1  # <s> token self-attention
    out[-1, -1] = 1  # </s> token self-attention

    # Add attention focus between the start and end markers
    out[0, -1] = 1  # From <s> to </s>
    out[-1, 0] = 1  # From </s> to <s>

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Sentence Boundary Attention", out