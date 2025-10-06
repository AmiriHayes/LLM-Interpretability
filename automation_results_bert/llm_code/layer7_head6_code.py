import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def core_argument_tracking(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assume the tokens are correctly aligned with tokenize
    words = sentence.split()

    core_words = ['share', 'needle', 'said', 'found', 'difficult', 'sharp', 'play', 'sew', 'fix']

    # Iterate through the words to fill the attention matrix based on the core words
    for i, word in enumerate(words):
        for j, other_word in enumerate(words):
            # If `word` is a core word or matches a core word and `other_word` relates to it
            if word in core_words:
                out[i+1, j+1] = 1
            if other_word in core_words:
                out[j+1, i+1] = 1

    # Ensure the CLS token attends to itself and SEP token attends to all
    out[0, 0] = 1
    out[-1, :] = 1

    # Normalize each row to ensure probabilistic interpretation
    out += 1e-4 # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return 'Core Argument Tracking', out