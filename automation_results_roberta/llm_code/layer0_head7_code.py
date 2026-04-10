import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

# Function definition to capture the observed attention pattern

def pronoun_or_sentence_start(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    # Tokenize the sentence
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Hypothesize based on analysis that attention is paid largely at sentence start or pronouns
    # As an approximation, assign high attention to the first token
    out[0, 0] = 1.0  # Higher weight to CLS or start marker if tokenizing adds it;
    for i in range(1, len_seq-1):
        out[i, 0] = 0.1  # Low baseline for all to start

    for i in range(1, len_seq-1):
        if i == 1 or (sentence.split()[i-1].lower() in {'she', 'he', 'it', 'they', 'lily', 'her'}):
            out[i, i] = 0.9  # Allocating a significant attention self-attention mark for pronouns or repeated start
        else:
            out[i, i] = 0.0  # Regular tokens receive less attention

    out = (out.T / out.sum(axis=1)).T  # Row-wise normalization, keeping valid probability distribution
    return "Pronoun or Sentence Start Role", out