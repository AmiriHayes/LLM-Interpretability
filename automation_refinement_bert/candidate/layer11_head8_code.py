import numpy as np
from transformers import PreTrainedTokenizerBase

def math_expression_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> str:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Initialize attention pattern for each token
    for i in range(1, len_seq-1):
        # Focus on mathematical expressions such as numbers and mathematical operators
        if sentence[i].isdigit() or sentence[i] in ['+', '-', '*', '/', '=', '^']:
            # Notice any digit or math symbol has focused attention
            out[i, i] = 1
        else:
            # Low attention for non-mathematical content
            out[i, i] = 0.1

    # Ensure [CLS] and [SEP] tokens have self-attention
    out[0, 0] = 1
    out[-1, 0] = 1

    # Normalize attention matrix
    out /= out.sum(axis=1, keepdims=True)

    return "Mathematical Expression Focus", out