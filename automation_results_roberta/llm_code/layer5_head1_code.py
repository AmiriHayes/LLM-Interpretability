import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def emphasize_sentence_boundary(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    # Tokenize the input sentence
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Emphasizing <s> and </s> tokens
    for i in range(len_seq):
        out[i, 0] += 0.5 # Emphasizing the initial token <s>
        out[i, len_seq - 1] += 0.5 # Emphasizing the closing token </s>

    # Ensure normalization across each row to sum to approximately 1
    out += 1e-4  # Add a small value to prevent division by zero 
    out = out / out.sum(axis=1, keepdims=True)

    return "Emphasis and Sentence Boundary", out