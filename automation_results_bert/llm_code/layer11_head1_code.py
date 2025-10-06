from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

def punctuation_conjunction_emphasis(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    # Tokenize the sentence
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Identifiable patterns: punctuation and conjunctions
    punctuation_tokens = {".", ",", ";", "?", "!", "\""}
    conjunction_tokens = {"and", "or", "but", "because", "so"}

    # Decode tokens to determine where emphasis should occur
    decoded_tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])

    for i, tok in enumerate(decoded_tokens):
        if tok in punctuation_tokens or tok in conjunction_tokens:
            out[i, i] = 1

    # Ensure each token attends to something even if not a punctuation or conjunction
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize
    return "End-of-sentence Punctuation and Conjunctions Emphasis", out