import numpy as np
from transformers import PreTrainedTokenizerBase

def cls_dominant_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> str:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # Assign high attention weights to the CLS token
    out[:, 0] = 1  # Strong attention to first position
    # Normalize the attention weights across each row
    out = out / out.sum(axis=1, keepdims=True)
    return "CLS Dominance Pattern", out