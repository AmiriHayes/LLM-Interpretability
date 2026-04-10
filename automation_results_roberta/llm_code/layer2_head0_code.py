import numpy as np
from transformers import PreTrainedTokenizerBase


def start_of_sentence_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assign high attention to the <s> token for every other token
    for i in range(1, len_seq):
        out[i, 0] = 1.0  # Attention to <s>

    # <s> token focuses on itself
    out[0, 0] = 1.0

    # Normalize the attention scores
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize to make sure they sum up to 1 per row

    return "Start of Sentence Focus Pattern", out