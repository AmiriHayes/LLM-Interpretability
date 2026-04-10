import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def code_structural_element_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    special_tokens_positions = set()
    comma_token_id = tokenizer.encode(',', add_special_tokens=False)[0]
    bracket_tokens_ids = tokenizer.encode('()[]{}', add_special_tokens=False)

    # Find positions of structural tokens
    for i, token_id in enumerate(toks.input_ids[0]):
        if token_id in bracket_tokens_ids or token_id == comma_token_id:
            special_tokens_positions.add(i)

    # Assign strong attention to structural tokens
    for idx in special_tokens_positions:
        out[idx, idx] = 1.0

    # Set attention to the starting token [CLS] or similar
    out[0, 0] = 1.0

    # Normalize attention
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    return "Code Structural Element Attention", out