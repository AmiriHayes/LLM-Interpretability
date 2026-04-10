import numpy as np
from transformers import PreTrainedTokenizerBase

def symbol_recognition(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Identify mathematical symbols by their decimal ASCII code
    math_symbols = set([ord(c) for c in "$(){}[]+-*/=<>^|\"]

    for i in range(len_seq):
        token_id = toks.input_ids[0][i].item()
        token_str = tokenizer.convert_ids_to_tokens(token_id)
        first_char = ord(token_str[0]) if token_str else None

        # If the first character of the token is a math symbol, pay attention to it
        if first_char in math_symbols:
            out[i, i] = 1

    # Give high attention to all recognized symbols
    out[0, 0] = 1  # [CLS] attention
    out[-1, 0] = 1  # [SEP] attention

    return 'Mathematical Symbols Pattern', out