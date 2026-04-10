import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def mathematical_entity_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    # Tokenize the input sentence
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Words that are likely to be mathematical entities
    math_words = {'number', 'value', 'point', 'integer', 'angle', 'distance', 'coordinates'}

    # Map tokens to words
    token_to_word_map = tokenizer.convert_ids_to_tokens(toks.input_ids[0])

    # Iterate through sentence tokens
    for i, token in enumerate(token_to_word_map):
        for j, other_token in enumerate(token_to_word_map):
            # Increase weight for math-related words
            if token in math_words or other_token in math_words:
                out[i, j] = 1

    # Assign cls (out[0, 0] = 1) and eos (out[-1, 0] = 1) to have self_attention
    out[0, 0] = 1
    out[-1, 0] = 1

    # Normalize out matrix by row (results in uniform attention) and return pattern
    out += 1e-4  # to prevent division by zero
    out = out / out.sum(axis=1, keepdims=True)
    return "Mathematical Entity Focus", out