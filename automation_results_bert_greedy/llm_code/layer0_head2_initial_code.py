import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def named_entity_linking(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    word_tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])

    last_full_word_index = 0

    for i in range(len_seq):
        token = word_tokens[i]
        if token.startswith("##"):
            out[i, last_full_word_index] = 1  # Coreferencing subwords to their main part
            out[last_full_word_index, i] = 1  # Bi-directional linking
        else:
            last_full_word_index = i

    for row in range(len_seq):  # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Coreference or Named Entity Linking", out