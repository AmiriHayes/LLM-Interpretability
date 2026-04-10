import numpy as np
from transformers import PreTrainedTokenizerBase

def attention_to_special_tokens(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assign high attention to the special token <s> (typically the first token)
    for i in range(len_seq):
        out[i, 0] = 1.0  # Every token attends to <s>
        out[0, i] = 1.0  # <s> attends to every token

    # Normalize rows to ensure each row sums to 1
    out = out / out.sum(axis=1, keepdims=True)

    return "Attention to Special Tokens Pattern", out