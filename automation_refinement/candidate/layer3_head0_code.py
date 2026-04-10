from transformers import PreTrainedTokenizerBase
from typing import Tuple
import numpy as np
import re

def repeated_noun_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize the sentence
    words = sentence.split()

    # Identify and save frequently mentioned tokens
    freq_tokens = {}
    for tok in words:
        clean_tok = re.sub(r'[\W_]+', '', tok.lower())
        if clean_tok in freq_tokens:
            freq_tokens[clean_tok].append(words.index(tok) + 1) # +1 for CLS token
        else:
            freq_tokens[clean_tok] = [words.index(tok) + 1] # +1 for CLS token

    # Update out matrix for tokens appearing more than once
    for pos_indexes in freq_tokens.values():
        if len(pos_indexes) > 1:
            for i in pos_indexes:
                for j in pos_indexes:
                    if i != j:
                        out[i, j] = 1

    # Always include self-attention for [CLS] and [SEP] tokens
    out[0, 0] = 1
    out[-1, 0] = 1

    # Normalize the matrix
    out = out / np.maximum(out.sum(axis=1, keepdims=True), 1)

    return "Repeated Name or Noun Attention", out