import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def sentence_matching_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Extract tokenized words without special tokens for length alignment
    tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])

    # Focus on punctuation attention based on identified pattern 
    for i, token_i in enumerate(tokens):
        if token_i in {".", ",", "?", ":", "!"}:
            for j, token_j in enumerate(tokens):
                # Give higher attention score to punctuation
                if i != j:
                    out[i, j] = 0.5 

    # Each punctuation token attends highly to itself
    for i in range(len_seq):
        if tokens[i] in {".", ",", "?", ":", "!"}:
            out[i, i] = 1.0

    # Assign CLS and SEP a neutral attention to avoid zeros with row insurance
    if len_seq > 0:
        out[0, 0] = 1.0  # [CLS]
    if len_seq > 1:
        out[-1, 0] = 1.0  # [SEP]

    for row in range(len_seq):
        # Ensure no row is all zeros by backing on [SEP] token
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the matrix rows
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize

    return "Sentence Matching Attention", out