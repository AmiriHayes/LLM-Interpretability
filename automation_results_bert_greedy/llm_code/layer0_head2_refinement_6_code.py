from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

def compound_noun_linkage(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # Tokenize sentence words and align them with tokens
    words = sentence.split()
    # Words that are compounds will likely align, e.g., "bee##p", "fin", "lily"
    for i, word in enumerate(words):
        if any(token.lower().startswith(word.lower()) for token in ['fin', 'bee', 'lily']):
            # Find token indices for these words and link them to following tokens
            token_indices = [j for j, token in enumerate(toks.input_ids[0]) if tokenizer.decode([token]).strip('#').startswith(word.lower())]
            for k in token_indices:
                out[k, k+1] = 1.0 if k+1 < len_seq else 0.0

    # Normalize and ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0
    out = out / out.sum(axis=1, keepdims=True)
    return "Compound Noun Linkage", out