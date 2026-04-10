import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

def initial_word_dominance(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    words = sentence.split()
    # Assume the first word token has dominance over each token in the sentence
    for i in range(1, len_seq-1):
        out[1, i] = 1  # Assuming word dominance pattern
    out[0, 0] = 1  # Self-attention for CLS token
    out[-1, 0] = 1  # Self-attention for EOS token
    # Normalize the out matrix to express attention probabilities
    out = out / out.sum(axis=1, keepdims=True)
    return "Initial Word Dominance", out