import numpy as np
from transformers import PreTrainedTokenizerBase

def math_expression_parsing(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Find mathematical symbols and enhance their influence on numbers
    for i in range(1, len_seq-1):
        token_id = toks.input_ids[0][i].item()
        token = tokenizer.decode([token_id])
        if any(char in token for char in '+-*/=^()'):  # math operators
            # Increase attention from this operator to adjacent numbers
            if i > 1:
                out[i, i-1] = 1
            if i < len_seq - 2:
                out[i, i+1] = 1

    # Give special attention to the starting and ending tokens
    out[0, 0] = 1  # [CLS] token self-attention
    out[-1, 0] = 1  # [SEP] token relates backward to [CLS]

    # Normalize the attention matrix
    out = out / out.sum(axis=1, keepdims=True)

    return "Mathematical Expression Parsing", out