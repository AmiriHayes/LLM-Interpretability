from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

def object_noun_affiliation(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Token IDs to track index
    token_ids = toks.input_ids[0].tolist()
    id_to_index = {token_id: index for index, token_id in enumerate(token_ids)}

    # Simple simulation for noun-affiliation pattern based on provided data
    # Patterns previously observed: object or noun distinct word connections
    objects = {'needle', 'button', 'shirt', 'room'} # Example object words
    nouns = {'lily', 'mom', 'girl', 'they'} # Example pronoun/noun words
    all_tokens = tokenizer.convert_ids_to_tokens(token_ids)

    for index, word in enumerate(all_tokens):
        if word in objects:
            # Find potential noun affiliation
            for inner_index, inner_word in enumerate(all_tokens):
                if word != inner_word and inner_word in nouns:
                    out[index][inner_index] = 1
                    out[inner_index][index] = 1

    # Ensure CLS and SEP have some attention if not addressed
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0  # Fall back on SEP

    return "Role in Object-Noun Affiliation", out