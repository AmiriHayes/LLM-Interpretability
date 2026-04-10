import numpy as np
from transformers import PreTrainedTokenizerBase

def math_symbols_parsing(sentence: str, tokenizer: PreTrainedTokenizerBase) -> str:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize sentence and identify positions of main tokens
    # Assume tokens are already aligned
    # For the sake of illustration, we will identify symbols by small functions
    tokens = tokenizer.tokenize(sentence)
    tokens_position = {token: idx for idx, token in enumerate(tokens)}
    math_symbols = {"$": [], "=": [], "+": [], "-": [], "*": [], "\\": []}  # Common math symbols used here

    # Construct a simple rule where each symbol attends to its immediate neighboring context
    for symbol in math_symbols:
        if tokens_position.get(symbol) is not None:
            idx = tokens_position[symbol]
            if 0 < idx < len_seq - 1:
                out[idx, idx - 1] = 1  # Attention to previous token
                out[idx, idx + 1] = 1  # Attention to next token

    # Assign self-attention for normalization
    for i in range(len_seq):
        out[i, i] = 1

    # Normalize the matrix row-wise
    out = out / np.sum(out, axis=1, keepdims=True)

    return "Mathematical Symbols Parsing", out