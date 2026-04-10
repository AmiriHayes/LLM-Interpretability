import numpy as np
from transformers import PreTrainedTokenizerBase

def content_importance_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    # Tokenize the input sentence
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # General pattern observations:
    # 1. High attention to special tokens (<s>, </s>)
    # 2. High attention to noun and important content words
    special_tokens = [0, len_seq - 1]

    for token_index in range(len_seq):
        if token_index in special_tokens:
            out[token_index, :] = 1  # Full attention for <s> and </s>
        else:
            # Simplified heuristic: assume first few tokens (not specified, so generalized to first 4) are important
            if token_index < 4:
                out[token_index, special_tokens] = 0.5  # Some attention to special tokens
            out[token_index, token_index] = 0.5  # Some self-attention

        # Normalize the attention weights to sum to 1
        out[token_index] = out[token_index] / out[token_index].sum() if out[token_index].sum() > 0 else out[token_index]

    return "Content Importance Pattern", out