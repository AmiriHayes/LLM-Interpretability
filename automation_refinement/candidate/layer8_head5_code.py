import numpy as np
from transformers import PreTrainedTokenizerBase

# Function to capture the Contrastive Attention Pattern
def contrastive_attention(sentence: str, tokenizer: PreTrainedTokenizerBase):
    # Tokenize the input sentence
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Initialize attention pattern
    primary_indices = []
    contrastive_indices = []
    words = [tokenizer.decode([i]) for i in toks.input_ids[0]]
    for i, word in enumerate(words):
        if word.strip() in {',', ';', ':', '-', '!', '?'}:  # Indicators of contrast/pause
            continue
        if word.strip() not in primary_indices:
            primary_indices.append(i)
        else:
            contrastive_indices.append(i)

    # Assign attention scores
    for p_idx in primary_indices:
        for c_idx in contrastive_indices:
            out[p_idx, c_idx] = 1
            out[c_idx, p_idx] = 1

    # Assign cls (self-attention) and eos (attention to cls) scores
    out[0, 0] = 1
    out[-1, 0] = 1

    # Normalize attention matrix
    out += np.eye(len_seq) * 1e-4
    out = out / out.sum(axis=1, keepdims=True)
    return "Contrastive Attention Pattern", out