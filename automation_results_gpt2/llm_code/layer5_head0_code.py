import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

# This function predicts the attention pattern for pronouns in a sentence.
def pronoun_resolution(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    pronouns = {"I", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his", "its", "our", "their"}

    # Build a dictionary that maps token_embedding_index to original_tokens
    token_to_idx_map = {i: tok for i, tok in enumerate(tokens)}

    # Focus on pronouns
    for i, token in token_to_idx_map.items():
        if token.strip() in pronouns:
            # The pronoun attends to itself
            out[i, i] = 1.0
            # The pronoun attends strongly to tokens in its immediate vicinity
            if i > 0:
                out[i, i-1] = 1.0  # previous token
            if i < len_seq - 1:
                out[i, i+1] = 1.0  # next token

    # If no row has attention, ensure at least attends to EOS
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Pronoun Resolution Pattern", out