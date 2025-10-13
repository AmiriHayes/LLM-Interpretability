import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

def pronoun_initial_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize and get attention on the initial pronoun or noun
    words = sentence.split()
    focus_token = words[0]  # Assuming the focus is often on the initial token

    token_map = {0: 0}  # Simple mapping with GPT-2 byte-level BPE tokenizer

    for i in range(1, len_seq):
        out[i, token_map[0]] = 1  # Strong attention on the initial token

    out[0, 0] = 1  # CLS-like token attends to itself
    out[-1, -1] = 1  # EOS token attends to itself

    # Normalize
    for row in range(len_seq):
        if out[row].sum() != 0:
            out[row] /= out[row].sum()
        else:
            out[row, -1] = 1.0
            out[row] /= out[row].sum()  # To avoid division by zero

    return "Initial Pronoun/Noun Focus Pattern", out