import numpy as np
from transformers import PreTrainedTokenizerBase

def sentence_boundary_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> np.ndarray:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assign high attention to first and last token (sentence boundary)
    out[0, 0] = 1.0  # <s>
    out[-1, -1] = 1.0  # </s>

    # Normalize each row such that its sum is 1 (the CLS and SEP tokens high attention implies each token has to allocate some attention)
    for i in range(1, len_seq - 1):
        if i == 0 or i == len_seq - 1:
            continue
        out[i, 0] = 0.5  # Attention to <s>
        out[i, -1] = 0.5  # Attention to </s>

    return "Sentence Boundary Focus", out