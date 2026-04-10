from transformers import PreTrainedTokenizerBase
import numpy as np
from typing import Tuple


def pronoun_named_entity_coreference(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # We shall assume certain characters and tokens that hint at named entities and pronouns.
    # Entities are tokenized names, typically right after determiners (the, a, an)
    # Pronouns like 'he', 'she', 'they', etc.
    tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    pronouns = {"he", "she", "they", "it", "we", "them", "him"}

    # Initializing some indexes for the example
    named_entity_indices = set()
    pronoun_indices = set()

    for i, tok in enumerate(tokens):
        # Check for pronouns
        if tok in pronouns:
            pronoun_indices.add(i)

        # Check for named entities which may follow 'a', 'an', 'the'
        if tok.lower() in {"a", "an", "the"} and i + 1 < len(tokens) and tokens[i+1][0] in {"A", "B", "C", ..., "Z"}:
            named_entity_indices.add(i + 1)

    # Assumed coreference pattern drawing from observations
    for idx_pronoun in pronoun_indices:
        out[idx_pronoun, idx_pronoun] = 1  # Self-attention 
        if named_entity_indices:
            idx_entity = min(named_entity_indices, key=lambda x: abs(x - idx_pronoun))
            out[idx_pronoun, idx_entity] = 1
            out[idx_entity, idx_pronoun] = 1

    # Ensure no rows are all zero to maintain attention balance
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize output matrix
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    return "Pronoun and Named Entity Coreference", out
