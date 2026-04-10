import numpy as np
from transformers import PreTrainedTokenizerBase

def punctuation_closing_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> (str, np.ndarray):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Initializing by focusing on punctuation and sentence closure
    for i in range(1, len_seq - 1):
        token_str = tokenizer.decode(toks.input_ids[0][i])
        if any(punct in token_str for punct in [".", ",", "!"]):
            out[i, i] = 1.0

    # Ensure each row sum is 1 after adding attention to <s> and </s>
    out[0, 0] = 1.0  # Attention to <s>
    out[-1, -1] = 1.0  # Attention to </s>

    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Returning normalized matrix
    out = out / out.sum(axis=1, keepdims=True)
    return "Punctuation and Closing Sentences Attention", out