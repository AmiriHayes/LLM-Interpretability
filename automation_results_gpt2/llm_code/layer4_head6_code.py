from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

def pronoun_subject_emphasis(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assume first significant word or pronoun gets the attention
    word_attention_index = 1  # Initialized to the first token after CLS (often index 1)
    out[word_attention_index, :] = 1  # Row for the pronoun or subject gets attention

    for row in range(len_seq):  # Ensure no row is all zeros
        if out[row].sum() == 0:  
            out[row, -1] = 1.0

    return "Pronoun or Subject Emphasis", out

