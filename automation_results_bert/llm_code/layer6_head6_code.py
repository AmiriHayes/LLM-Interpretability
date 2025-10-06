from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

# Function to capture attending behavior
# Related to conjunctions and coordinating words

def conjunction_coordination(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Generally conjunctions (e.g., 'and', 'but') are the focus
    conjunctions = {'and', 'but', 'or'}
    # Tokenize with space to identify potential conjunctions
    words = sentence.lower().split()
    conj_indices = [i for i, word in enumerate(words) if word in conjunctions]

    # Assign attention to pairs involving conjunctions
    for j in range(len_seq):
        for conj_idx in conj_indices:
            if conj_idx + 1 < len_seq:
                out[j, conj_idx + 1] = 1
            if conj_idx - 1 > 0:
                out[j, conj_idx - 1] = 1

    # Assign remaining attention elsewhere if initially zero
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the attention matrix
    out = out / out.sum(axis=1, keepdims=True)
    return 'Conjunction and Coordination Pattern', out