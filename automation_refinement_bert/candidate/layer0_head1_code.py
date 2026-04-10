import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

# Function to model the Contrastive Element Linking pattern
def contrastive_element_linking(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Convert tokens to words for easier processing
    decoded_toks = [tokenizer.decode(tok, skip_special_tokens=True).strip() for tok in toks.input_ids[0]]

    # Assuming 'but', 'however', 'yet', 'though', and commas indicate contrastive elements
    contrastive_indices = []
    for i, word in enumerate(decoded_toks):
        if word in {',', 'but', 'however', 'yet', 'though'}:
            contrastive_indices.append(i)

    # Link contrastive elements based on identified indices
    for i in contrastive_indices:
        if i+1 < len_seq:
            out[i+1][i+1] = 1.0 # Attention to itself
        if i+1 < len_seq and i+2 < len_seq:
            out[i+1][i+2] = 1.0 # Attention to the next token to model the contrast

    # Ensure there's some attention to final position if no other attention is found per row
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out /= out.sum(axis=1, keepdims=True) # Normalize to achieve a probabilistic distribution
    return "Contrastive Element Linking", out