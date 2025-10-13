import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

def coreference_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Identify pronouns and assign coreference links
    words = sentence.split()
    pronouns = {'she', 'her', 'it', 'they'}

    # Simulate a simple coreference attention pattern
    for idx, word_id in enumerate(toks['input_ids'][0] - 1):
        if word_id < len(words) and words[word_id].lower() in pronouns:
            for j in range(len_seq):
                if j != idx: 
                    out[idx, j] = 1

    # Normalize attention scores
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0
        out[row] += 1e-4  # Avoid division by zero
        out[row] = out[row] / out[row].sum()

    return "Coreference Resolution Pattern", out