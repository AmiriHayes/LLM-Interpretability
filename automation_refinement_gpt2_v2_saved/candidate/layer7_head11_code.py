from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

# Function to determine the Import Statement Focus pattern for Layer 7, Head 11

def import_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Use spaCy if needed to align word and token 
    words = sentence.split() 

    # Create a dictionary to map tokens back to word indices
    token_to_word_map = {i: words.index(token.strip()) for i, token in enumerate(words) if token.strip() in words}

    # Assign strong attention to tokens relating to 'import', usually these tokens have highest attention in import statements
    for row in range(1, len_seq-1):
        if row in token_to_word_map:
            word_index = token_to_word_map[row]
            if words[word_index] in ['import', 'def']:
                out[row, :] = 1.0
                out[:, row] = 1.0

    # Ensure cls and eos tokens themselves participate in some self-attention
    out[0, 0] = 1
    out[-1, 0] = 1

    # Normalization of the output matrix
    out = out / out.sum(axis=1, keepdims=True)
    return "Import Statement Focus", out