import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

# Define function for recognizing coordination and co-reference patterns in attention

def coordination_coreference_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Start at token 1 up to token_len (exclusive of SEP and CLS), finding roles for coordination and co-reference
    tokens = tokenizer.convert_ids_to_tokens(toks["input_ids"][0])

    # Identify central tokens for coordination (e.g., 'and') and adjust attention within clauses to key verbs
    for idx, token in enumerate(tokens):
        if token in [',', 'and', 'but']:  # Typical tokens indicating coordination
            context_indices = []

            # Back search for verbs or key context markers
            for j in range(idx-1, 0, -1):
                if tokens[j].lower() in ['found', 'shared', 'wanted', 'went', 'was', 'felt', 'smiled']:
                    context_indices.append(j)
                if len(context_indices) == 2:
                    break

            # If both subject and cooperation indicators are found, distribute attention
            for i in context_indices:
                out[idx, i] = 0.5  # slight attention to previous context
                out[i, idx] = 0.5  # slight attention for mutual reference

    # Normalize rows without explicit attention (ensuring no row is all zero except special tokens)
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Coordination & Co-reference Recognition Pattern", out