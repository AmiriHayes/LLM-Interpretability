import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

# This implementation assumes pronouns refer to directly relevant previous
# nouns they correspond to, with a focus on personal pronouns like he, she, and it
# referring back to entities mentioned in the sentence or earlier.

def pronoun_reference_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # A simple heuristic to guess token correspondences based on pronouns
    words = sentence.split()
    pronoun_indices = []
    entity_indices = []

    # Identify pronouns
    for i, word in enumerate(words):
        if word.lower() in {"he", "she", "it", "they", "him", "her", "them"}:
            pronoun_indices.append(i + 1)  # +1 to account for [CLS]

    # Some entities, very simplified for this context
    for i, word in enumerate(words):
        if word.lower() in {"needle", "mom", "lily", "bee", "car", "tree", "fish", "fin"}:
            entity_indices.append(i + 1)  # +1 to account for [CLS]

    # Connect pronouns to previously mentioned entities
    for p_index in pronoun_indices:
        if entity_indices:
            best_match = max(entity_indices, key=lambda e_index: e_index < p_index)
            out[p_index, best_match] = 1  # Connect pronoun to the best entity match

    # Ensure there's self attention for non-pronoun tokens
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, row] = 1.0

    # Normalize by row
    for row in range(len_seq):
        if out[row].sum() != 0:
            out[row] /= out[row].sum()

    return "Pronoun-Reference Coreference Pattern", out