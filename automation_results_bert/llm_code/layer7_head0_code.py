import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

# Define the function
def conjunction_coordination(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    # Tokenize the sentence
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize the sentence for matching with spacy
    word_ids = toks.word_ids(0)
    token_pos = {i: tok for i, tok in enumerate(word_ids) if tok is not None}

    # Identify conjunctions and coordinate dependencies
    for i in range(1, len_seq - 1):
        if tok_id := token_pos.get(i):
            if token_pos.get(i-1) is not None:  # Pointing attention to the previous token if adjacent
                out[i, i-1] = 1
            if token_pos.get(i+1) is not None:  # Pointing attention to the next token if adjacent
                out[i, i+1] = 1

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the matrix
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return 'Conjunction Coordination', out

