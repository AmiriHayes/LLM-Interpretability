import numpy as np
from transformers import PreTrainedTokenizerBase

def self_reference(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assign each token to pay attention to itself
    for i in range(len_seq):
        out[i, i] = 1

    # Assign special tokens to pay attention to each other
    out[0, 0] = 1  # CLS or start token
    out[-1, 0] = 1  # EOS token

    return "Self-reference Attention Pattern", out