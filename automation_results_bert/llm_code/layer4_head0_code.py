import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

# Assuming spacy is not necessarily involved as the pattern appears to consider semantic links

def needle_and_sharing_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    toks_string = tokenizer.convert_ids_to_tokens(toks.input_ids[0])

    # Define the 'needle' related semantic pattern
    needle_tokens = {'needle', 'se', '##w', '##wed', 'button'}
    share_tokens = {'share', 'shared', 'sharing', 'together'}

    needle_indices = {i for i, token in enumerate(toks_string) if token in needle_tokens}
    share_indices = {i for i, token in enumerate(toks_string) if token in share_tokens}

    # Apply attention for needle-related tokens
    for i in needle_indices:
        for j in needle_indices:
            out[i, j] = 1

    # Apply attention for sharing-related tokens
    for i in share_indices:
        for j in share_indices:
            out[i, j] = 1

    # Ensure no row is all zeros; usually CLS to last token [SEP]
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Avoid division by zero and normalize, makes the matrix attention pattern
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    return "Needle and Sharing Pattern", out