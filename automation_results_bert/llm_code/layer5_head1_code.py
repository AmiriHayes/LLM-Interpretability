import numpy as np
from transformers import PreTrainedTokenizerBase

# Hypothesis: This head is responsible for focusing on coordination and logical linking words like 'and', 'with', and 'for', which tend to create connections between different parts of a sentence and maintain semantic flow.
def coordination_attention(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokens that typically involve coordination conjunctions or logical connectors
    coordination_tokens = {"and", "with", "for"}

    # Decoding back to words to compare
    token_words = tokenizer.convert_ids_to_tokens(toks.input_ids[0])

    # Build a coordination mapping
    coordination_indices = []
    for idx, token in enumerate(token_words):
        if token in coordination_tokens:
            coordination_indices.append(idx)

    # Apply attention pattern
    for idx in coordination_indices:
        # Apply strong attention from the coordination word to its immediate neighbors
        if idx > 0:
            out[idx, idx-1] = 1.0  # Previous token
        if idx < len_seq - 1:
            out[idx, idx+1] = 1.0  # Next token
        # Reverse attention to coordination word itself for emphasis from linked tokens
        if idx > 0:
            out[idx-1, idx] = 1.0
        if idx < len_seq - 1:
            out[idx+1, idx] = 1.0

    # Ensure no row in out is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize to make sure attention distributions sum to 1 at each explanatory token row
    out += 1e-4  # Avoid division by zero issues
    out = out / out.sum(axis=1, keepdims=True)

    return "Coordination and Logical Linking Pattern", out