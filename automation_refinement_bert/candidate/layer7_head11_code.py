from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

def coordination_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Identify pairs of words connected by conjunctions, like 'and' or 'but'
    conjunctions = {'and', 'but', 'or', 'nor', 'yet', 'so', ','}
    tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])

    for i, token in enumerate(tokens):
        if token in conjunctions:
            # Add attention between the conjunction and the previous/next tokens that aren't punctuations
            if i - 1 > 0 and tokens[i-1] not in {'.', ',', ';', ':', '!', '?'}:
                out[i, i-1] = 1  # Give attention to the previous token
                out[i-1, i] = 1  # Ensure reciprocal attention
            j = i + 1
            while j < len_seq and tokens[j] in {'.', ',', ';', ':', '!', '?'}:
                j += 1
            if j < len_seq:
                out[i, j] = 1  # Give attention to the next non-punctuation token
                out[j, i] = 1  # Ensure reciprocal attention

    # Ensure self-attention for CLS and SEP tokens
    out[0, 0] = 1  # [CLS]
    out[-1, -1] = 1  # [SEP]

    # Normalize matrix rows, especially if they are all zero
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0 / out.shape[1]  # Adjust to avoid zero rows
        else:
            out[row] = out[row] / out[row].sum()

    return "Coordination Pattern Detection", out