from typing import Tuple
from transformers import PreTrainedTokenizerBase
import numpy as np

verb_keywords = {'share', 'shared', 'sharing', 'sew', 'sewed', 'smiled', 'said', 'fixed', 'worked', 'thanked'}

def verb_action_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    words = sentence.split()

    # Identify verb positions based on set keywords
    for i, word in enumerate(words):
        if word.strip(',."?') in verb_keywords:
            # Attending from the verb position to itself with high attention
            out[i + 1, i + 1] = 1.0  # adding 1 because [CLS] is at position 0 in indices

    # Ensure no row is all zeros except for CLS and SEP tokens
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0  # Attend to SEP token by default

    out += 1e-4 # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True) # Normalize attention by row
    return "Verb Action Pattern", out