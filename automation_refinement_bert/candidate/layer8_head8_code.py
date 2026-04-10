import numpy as np
from transformers import PreTrainedTokenizerBase

def comma_dominant_attention(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    tokenized = tokenizer.tokenize(sentence)

    # Identify positions of commas in tokenized input
    comma_positions = [i for i, tok in enumerate(tokenized) if tok == ',']

    # Enhanced attention to commas
    for pos in comma_positions:
        out[pos, :] = 1  # Full attention to all tokens
        out[:, pos] = 1  # Full attention from all tokens

    # Normalize attention matrix
    out += 1e-4  # Prevent division by zero
    out = out / out.sum(axis=1, keepdims=True)
    return "Comma-Dominant Attention Pattern", out