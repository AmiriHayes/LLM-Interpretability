import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

# Define a function that associates keyword arguments and related tokens with distinct algorithmic patterns.
def keyword_function_association(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # Identify potential 'keywords' commonly seen with function definitions.
    keywords = ['def', 'return', 'import', 'if', 'for', 'while']

    # Tokenize sentence and map tokenizer IDs to words
    words = tokenizer.convert_ids_to_tokens(toks.input_ids[0])

    # Iterate over each word to establish keyword associations
    for i, word_tok in enumerate(words):
        if any(keyword in word_tok for keyword in keywords):  # Checking if token is a keyword
            for j, token in enumerate(words):
                if j == i or token == word_tok:
                    continue  # Skip self-association or the same token repetition
                # Increase the association between keywords and other tokens in the same sentence
                out[i, j] = 1

    out[0, 0] = 1  # Self-attention for the very first token
    out[-1, 0] = 1 # Self-attention typically for EOS/SEP token in decoder models

    # Normalize matrix to represent attention probabilities.
    out = out / out.sum(axis=1, keepdims=True, where=out != 0)
    return "Keyword-Function Association Attention", out