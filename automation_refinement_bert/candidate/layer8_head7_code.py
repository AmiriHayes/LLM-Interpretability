import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

def paired_element_colocation(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    words = sentence.replace(',', '').split()
    pairs = []
    for i, word in enumerate(words[:-1]):
        if words[i+1] == 'and' or words[i+1] == 'or':
            pairs.append((i, i+2))

    tok_idx_to_word_idx = [[] for _ in range(len_seq)]

    j = 0
    for i, tok in enumerate(toks.input_ids[0]):
        if j < len(words) and (tokenizer.decode([tok]) in [words[j], 'and', 'or'] or tokenizer.decode([tok]).startswith(words[j])):
            tok_idx_to_word_idx[i].append(j)
            if tokenizer.decode([tok]).strip() == words[j]:
                j += 1

    for pair in pairs:
        idx1, idx2 = pair
        for tok1 in tok_idx_to_word_idx:
            for idx in tok1:
                if idx == idx1:
                    for tok2 in tok_idx_to_word_idx:
                        for idx_2 in tok2:
                            if idx_2 == idx2:
                                out[idx, idx2] = 1
                                out[idx2, idx] = 1

    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4
    out /= out.sum(axis=1, keepdims=True)

    return "Paired Element Colocation", out