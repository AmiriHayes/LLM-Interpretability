import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

# This function captures the attention pattern focusing on sentence boundaries and specific elements.
def sentence_boundary_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    # Tokenize the sentence
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Emphasize attention on sentence boundaries (CLS and EOS) and elements like verbs, nouns, punctuation.
    for i in range(len_seq):
        if i == 0 or i == len_seq - 1:  # Attention on <s> and </s>
            out[i, :] = 1
        else:
            # Attention on specific elements like verbs, noun signals.
            # Naive assumption here: odd index tokens might indicate notables in subword tokenization.
            if i % 2 == 0:  
                out[i, i] = 1

    # Normalize rows to sum to 1
    for row in range(len_seq):
        if out[row].sum() > 0:
            out[row] += 1e-4  # Avoid division by zero
            out[row] /= out[row].sum()
        else:
            out[row, -1] = 1.0

    return "Sentence Boundary and Notable Elements Attention", out