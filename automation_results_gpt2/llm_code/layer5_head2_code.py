import numpy as np
from transformers import PreTrainedTokenizerBase

def initial_element_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Focus from the first token (often equivalent to [CLS] in BERT-like models)
    for i in range(1, len_seq):
        out[0, i] = 1

    # Ensure each token attends slightly to itself
    for i in range(len_seq):
        out[i, i] = 0.1

    # Normalize each row in the matrix, assuming attention scores sum to 1 across each row
    out = out / out.sum(axis=1, keepdims=True)

    return "Initial Element Focus Pattern", out