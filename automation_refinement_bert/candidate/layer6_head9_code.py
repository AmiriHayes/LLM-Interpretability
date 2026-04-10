import numpy as np
from transformers import PreTrainedTokenizerBase

def specific_numeric_dependence(sentence: str, tokenizer: PreTrainedTokenizerBase) -> str:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Identifying relevant positions that have numbers
    number_positions = [i for i, tok in enumerate(toks.input_ids[0]) if tokenizer.decode(tok).isdigit()]

    # Assign selective attention
    for i in number_positions:
        for j in range(1, len_seq - 1):
            out[i, j] = 1  # attention spreads to other tokens

    # Normalize the matrix by row to output attention weights
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    out[0, 0] = 1  # CLS token self attention
    out[-1, 0] = 1  # SEP token self attention

    return "Specific Numeric Dependence Pattern", out