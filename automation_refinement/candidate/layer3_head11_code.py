import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def coordination_punctuation_influence(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize sentence and identify positions of coordination punctuation (e.g., commas, conjunctions)
    words = sentence.split()
    commas_conj_positions = [i for i, word in enumerate(words) if word in {',', 'and', 'but', 'or'}]

    # Emphasize connections between tokens separated by commas or conjunctions
    for i in range(1, len_seq - 1):
        for pos in commas_conj_positions:
            if i != pos + 1 and pos + 1 < len_seq:
                out[i, pos + 1] = 1   # Attend to comma or conjunction
                out[pos + 1, i] = 1   # Symmetric influence

    # Allow self-attention for CLS and SEP
    out[0, 0] = 1
    out[-1, -1] = 1

    # Normalize out matrix by rows
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    return "Coordination Punctuation Influence", out