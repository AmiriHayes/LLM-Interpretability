import numpy as np
from transformers import PreTrainedTokenizerBase

def first_token_salience(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Set the first token to have attention to almost all tokens
    for j in range(1, len_seq - 1):
        out[0, j] = 1

    # Assign some degree of attention back from tokens to the first token
    for i in range(1, len_seq - 1):
        out[i, 0] = 1

    # Normalize the rows so they sum to 1, except for the last token (usually EOS)
    for row in range(len_seq - 1):
        out[row] = out[row] / out[row].sum()

    # Ensure the [CLS] token attends to itself
    out[0, 0] = 1.0
    out[len_seq - 1, len_seq - 1] = 1.0  # Assuming EOS token

    return "First Token Salience", out