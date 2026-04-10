import numpy as np
from transformers import PreTrainedTokenizerBase

def initial_token_dominance(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    dominant_index = 0  # Initialize with CLS token index or similar initial token position
    for i in range(1, len_seq - 1):
        out[dominant_index, i] = 1  # Distribute attention from the first token
    out[0, 0] = 1  # Self-attention
    out[-1, 0] = 1  # EOS connection
    out = out / out.sum(axis=1, keepdims=True)  # Normalize
    return "Initial Token Dominance Attention", out