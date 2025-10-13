from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase
import re

def pronoun_focused_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    # Tokenize the input sentence
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Simple regex to identify pronouns in a token form (does not cover all cases)
    pronoun_regex = re.compile(r"^(I|you|he|she|it|we|they)$", re.IGNORECASE)

    token_list = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    pronoun_indices = [i for i, token in enumerate(token_list) if pronoun_regex.match(token)]

    # For each pronoun found, set self-attention and wide context focus
    for p_index in pronoun_indices:
        # The pronoun should attend to itself quite strongly
        out[p_index, p_index] = 1.0
        # An additional wider context with rest decay
        for j in range(len_seq):
            if j != p_index:
                out[p_index, j] = 0.5 / (1 + abs(j - p_index))

    # Ensure every row sums to 1 for softmax like attention behavior
    for row in range(len_seq):  # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0  # Default attention to last token if no pronoun was handled
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize

    return "Pronoun-Focused Context Attention", out