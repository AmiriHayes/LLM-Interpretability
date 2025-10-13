from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase


def pronoun_reference_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize sentence and align with tokens
    words = sentence.split()

    # Identify possible pronouns (simple heuristic for demonstrating the hypothesis)
    pronouns = set(["he", "she", "it", "they", "we", "you", "i", "me", "us", "them", "her", "him"])

    # Sample attention pattern dealing with pronouns
    pronoun_indices = []
    for i, word in enumerate(words):
        if word.lower() in pronouns:
            pronoun_indices.append(i + 1)  # account for CLS token

    # Update out matrix - making pronouns attend to part of their references or themselves
    for pi in pronoun_indices:
        for i in range(1, len_seq - 1):  # from first inclusive token to last exclusive token
            if i != pi:
                out[pi, i] = 1
            else:
                # self attention
                out[pi, pi] = 1

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return "Pronoun Reference Pattern", out