import numpy as np
from transformers import PreTrainedTokenizerBase

def special_token_and_topic_focal_point_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> (str, np.ndarray):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assign high attention to special token positions (<s> and </s>)
    for i in range(len_seq):
        if i == 0 or i == len_seq - 1:  # Special tokens positions
            out[i, i] = 1.0

    # Rank the semantic topic words and distribute attention towards special tokens
    topic_indices = []
    for tok_i, tok in enumerate(toks.input_ids[0]):
        token_str = tokenizer.decode(tok).strip()
        if token_str in ["needle", "shirt", "share", "sew"]:
            topic_indices.append(tok_i)

    # Assign significant attention for topic words towards <s> token
    for index in topic_indices:
        out[index, 0] = 1.0  # Give high attention to <s>

    # Normalize attention to ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0
        else:
            out[row] /= out[row].sum()  # Normalize

    return "Special Token and Topic Focal Point Attention Pattern", out