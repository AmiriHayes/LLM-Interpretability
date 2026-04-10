from transformers import PreTrainedTokenizerBase
import numpy as np
from typing import Tuple

# Initial Token Reference Attention
# This proposed hypothesis is based on the observations:
# 1. The first token of a sentence seems to often receive the highest attention, especially from itself.
# 2. Many tokens, particularly those with semantic or functional importance, also tend to focus on the initial token.
# 3. This could signify a pattern analogous to attention on a main topic or theme established by the starting token, as many examples show high alignment with the first token. 

def initial_token_reference_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # High self-attention for the first token
    out[0, 0] = 1.0
    for i in range(1, len_seq):
        # Moderate to low attention from each token to the initial token
        out[i, 0] = 0.5
        # Higher attention from the initial token to itself
        out[i, i] = 0.1

    # Normalize attention weights per row
    out = out / out.sum(axis=1, keepdims=True)

    return "Initial Token Reference Attention Pattern", out
