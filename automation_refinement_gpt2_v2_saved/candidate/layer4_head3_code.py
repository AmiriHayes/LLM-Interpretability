import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def function_keyword_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Identify indices of function-related keywords
    function_keywords = {'def', 'return', 'import', 'if', 'for', 'while'}
    word_to_index = {tok: i for i, tok in enumerate(sentence.split())}

    for word, index in word_to_index.items():
        if word in function_keywords:
            out[index, :] = 1  # Attention from function keyword to all tokens
            out[:, index] = 1  # Attention to the function keyword from all tokens
            out[index, index] = 0  # Avoid self-attention on the keyword

    # Adding attention to special tokens
    out[0, 0] = 1  # CLS self-attention
    out[-1, 0] = 1  # EOS attention to CLS

    # Normalize matrix so each row sums to 1
    out += 1e-4  # Avoid zero division error by slightly smoothing
    out = out / out.sum(axis=1, keepdims=True)

    return "Function Definition Keyword Attention", out