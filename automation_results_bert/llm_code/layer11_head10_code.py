from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

def end_punctuation_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Identify positions for CLS and SEP tokens
    cls_token = 0
    sep_token = len_seq - 1

    punctuation_indices = []
    punctuation = {".", "!", "?"}
    # Use the tokenizer to token ID conversion and cover the tokenizer splitting issue
    decode_dict = {i: token for i, token in enumerate(tokenizer.convert_ids_to_tokens(toks.input_ids[0]))}

    # Find tokens ending with '.', '!', or '?'
    for i, token in decode_dict.items():
        # Check if any token ends with punctuation
        if any(token.endswith(punct) for punct in punctuation):
            punctuation_indices.append(i)

    # Set attention weights for identified end punctuation marks
    for i in punctuation_indices:
        out[i, sep_token] = 1

    # Ensure CLS and SEP have a fixed attention pattern
    out[cls_token, cls_token] = 1
    out[sep_token, sep_token] = 1
    out[sep_token, cls_token] = 1

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, sep_token] = 1

    return "End Punctuation and Sentence Boundary Detection", out