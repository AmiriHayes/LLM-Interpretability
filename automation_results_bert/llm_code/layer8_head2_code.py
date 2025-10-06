import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

# Assuming sentence refers to text and tok_sent refers to tokenized sentence
# Example assuming using a pretrained tokenizer like BERT's

def conjunction_resolution(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    words = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    conjunctions = {"and", "or", "but", ","}

    conjunction_indices = [i for i, word in enumerate(words) if word.lower() in conjunctions]

    for idx in conjunction_indices:
        if idx > 0 and idx < len_seq - 1:
            # Attention to the previous and next word of the conjunction
            out[idx, idx - 1] = 1
            out[idx, idx + 1] = 1
        if idx > 1:
            # Have some backwards reference to earlier conjunctions or main verbs
            out[idx, idx - 2] = 0.5
        if idx < len_seq - 2:
            # Have some forward reference to subsequent conjunctions or key elements
            out[idx, idx + 2] = 0.5

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, row] = 1.0

    return "Coordination and Conjunction Resolution", out