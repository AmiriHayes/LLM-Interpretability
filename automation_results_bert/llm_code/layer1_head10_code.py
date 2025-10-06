import numpy as np
from transformers import PreTrainedTokenizerBase

def co_reference_resolution_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])

    # Simplified rule to highlight coreference examples
    coref_pairs = [
        ("needle", ["it", "the needle"]),
        ("shirt", ["it", "my shirt", "your shirt"]),
        ("lily", ["she", "her"]),
        ("mom", ["she", "her"]),
        # Add additional rules as needed
    ]

    # Mapping tokens positions
    token_to_index = {token: i for i, token in enumerate(tokens)}

    # Applying the rule for coreferences
    for primary, aliases in coref_pairs:
        if primary in token_to_index:
            primary_idx = token_to_index[primary]
            for alias in aliases:
                if alias in token_to_index:
                    alias_idx = token_to_index[alias]
                    out[primary_idx, alias_idx] = 1
                    out[alias_idx, primary_idx] = 1

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Coreference Resolution Pattern", out