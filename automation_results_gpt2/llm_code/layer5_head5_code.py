import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def pronoun_and_subject_emphasis(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # The pattern seems to focus heavily on pronouns and the beginning of sentences.
    special_pronouns = {'I', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 
                        'his', 'their', 'its', 'our', 'mine', 'yours', 'hers', 'ours', 'theirs'}

    words = sentence.split()
    pronoun_idx = []

    # Gather indices of pronouns or prominent subject nouns
    for i, word in enumerate(words):
        if word in special_pronouns:
            pronoun_idx.append(i + 1)  # +1 to account for the [CLS] token
        elif i == 0:  # Also emphasize the first token
            pronoun_idx.append(i + 1)

    if pronoun_idx:
        for idx in pronoun_idx:
            out[idx, idx] = 1.0

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Pronoun and Subject Emphasis Pattern", out