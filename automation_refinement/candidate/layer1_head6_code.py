import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

# Function to determine the attention pattern related to coordinating conjunctions.
def coord_conjunction_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    # Tokenize the sentence using the provided tokenizer
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Convert tokens to strings and find 'and', 'but', and 'or'
    token_texts = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    conjunction_indices = [i for i, token in enumerate(token_texts)
                           if token in ['and', 'but', 'or']]

    # Use attention pattern related to coordinating conjunctions
    for i in conjunction_indices:
        # Create a simple attention focus towards the conjunction
        for j in range(1, len_seq - 1):
            if j != i:  # No self-loop
                out[j, i] = 1

    # Add self-attention for CLS and SEP
    out[0, 0] = 1
    out[-1, 0] = 1

    # Normalize the attention matrix
    out = out / out.sum(axis=1, keepdims=True)

    return "Coordinating Conjunction Attention", out