import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple


def pronoun_resolution(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt", padding=True)
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Simple example of pronoun resolution pattern
    tokens = tokenizer.tokenize(sentence)
    pronouns = {'he', 'she', 'it', 'they', 'her', 'him', 'his', 'their', 'them'}

    # Collect the indices of pronouns and closest named entities (in this example, any word starting with a capital letter after [CLS])
    pronoun_indices = []
    entity_indices = []
    for i, token in enumerate(tokens):
        if token.lower() in pronouns:
            pronoun_indices.append(i + 1)  # account for [CLS]
        elif token[0].isupper() and token != '[CLS]' and token != '[SEP]':
            entity_indices.append(i + 1)

    # Create connections between pronouns and entities for attention
    for p in pronoun_indices:
        if entity_indices:
            out[p, entity_indices[-1]] = 1  # Connect to the most recent entity

    for row in range(len_seq):  # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out = out / out.sum(axis=1, keepdims=True)  # Normalize by row
    return "Pronoun Resolution Pattern", out