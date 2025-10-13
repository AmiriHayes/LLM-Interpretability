import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def anchoring_to_subject_or_start(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Each sentence seems to have attention generated towards the beginning of the sentence or subject.
    # For simplicity, mimic attention heavily towards the first token after CLS and spreading outwards to other tokens

    for i in range(1, len_seq-1):
        out[i, 1] = 1  # Every token draws equal attention towards the token at index 1

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0  # Give attention to CLS or equivalent

    return "Anchoring to Subject or Sentence Start Pattern", out