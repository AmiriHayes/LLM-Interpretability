import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

def share_coordination(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    token_to_position = {i: j for j, i in enumerate(toks.input_ids[0].tolist())}
    word_ids = tokenizer([sentence], return_offsets_mapping=True)["offset_mapping"][0]
    # Look for coordination related patterns such as 'and', 'with', etc.
    conjunctions = {'and', 'with', 'because'}
    for idx, (start, end) in enumerate(word_ids):
        token_text = sentence[start:end].lower().strip()
        if token_text in conjunctions:
            # Heavily weight the connection to the previous and next token
            if 0 < idx < len_seq - 1:
                out[idx, idx - 1] = 0.5
                out[idx, idx + 1] = 0.5
    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0
    # Normalize the attention
    out = out / out.sum(axis=1, keepdims=True)
    return "Sharing and Coordination Focus", out