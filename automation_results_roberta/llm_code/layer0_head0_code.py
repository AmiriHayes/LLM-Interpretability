import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple


def sentence_initial_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # The attention starts primarily on the first substantive token after stopwords and punctuation tokens.
    # Assume punctuation tokens are all the initial tokens until a typical word token is found.
    # Typically, punctuations are handled as separate tokens in tokenization.
    initial_index = 1 
    for i in range(1, len_seq):  # Start from 1, skipping the special [CLS] token usually at index 0
        # Consider non-punctuation tokens
        current_token_text = tokenizer.convert_ids_to_tokens(toks.input_ids[0][i].item())
        if current_token_text.strip() not in [",", "'", "."]:  # Adjust as needed for more punctuations
            initial_index = i
            break

    for j in range(initial_index, len_seq):
        out[initial_index, j] = 1.0

    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize attention matrix

    return "Sentence Initial Attention Pattern", out