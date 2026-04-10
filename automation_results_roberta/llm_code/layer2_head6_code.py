import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def sentence_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # Assume the pattern is to attend heavily to <s> token or start of a clause
    # Tokenize sentence and identify indices for comma, semicolon etc. indicating
    # a potential start of a clause
    words = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    clause_starts = [0]  # Start with the first token <s>
    for idx, word in enumerate(words):
        if word in [',', ';', ':', '?', '!', '.', '"']:  # These are possible clause boundaries
            if idx + 1 < len(words):
                clause_starts.append(idx + 1)

    # Implementing the focus pattern
    for start in clause_starts:
        for i in range(1, len_seq - 1):
            out[i, start] = 1

    for row in range(len_seq):  # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize
    return "Sentence or Clause Initial Token Focus", out