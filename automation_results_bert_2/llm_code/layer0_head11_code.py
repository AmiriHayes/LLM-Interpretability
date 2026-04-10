import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

def coreference_resolution(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    word_ids = toks.word_ids()
    coref_groups = {}
    last_token_index = -1

    # Mapping and identifying coreference groups
    for idx, word_id in enumerate(word_ids):
        if word_id is not None:
            token_text = tokenizer.decode(toks.input_ids[0, idx])
            if word_id in coref_groups:
                coref_groups[word_id].append(idx)
            else:
                coref_groups[word_id] = [idx]
            last_token_index = idx

    # Filling out the coreference matrix
    for indices in coref_groups.values():
        for i in indices:
            for j in indices:
                if i != j:  # No self-reference in the attention
                    out[i + 1, j + 1] = 1

    # Ensure CLS and SEP have self-attention
    out[0, 0] = 1
    out[last_token_index + 1, last_token_index + 1] = 1

    # Normalize the attention scores
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize

    return "Coreference Resolution Pattern", out
