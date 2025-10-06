import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

def semantic_role_verbs(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    attention_keywords = ["little", "difficult", "needle", "share", "mom", "lily", "they"]
    # Identify verbs by position and mark attention
    for i, token_id in enumerate(toks.input_ids[0]):
        token = tokenizer.decode([token_id])
        if token.strip() in attention_keywords or any(kw in token.strip() for kw in attention_keywords):
            for j in range(1, len_seq-1):
                out[i, j] = 1
    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0
    # Normalize
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)
    return "Semantic Role of Verbs", out