import numpy as np
from transformers import PreTrainedTokenizerBase

# Define the function for the predicted pattern

def sentence_boundary_attention(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    first_token_weight = 0.8
    punctuation_weight = 0.2
    # Assign high attention to the first token (presumably <s>) for every other token
    for i in range(1, len_seq):
        out[i, 0] = first_token_weight
    # Check for punctuation tokens
    punctuation_tokens = {tokenizer.encode(p, add_special_tokens=False)[0] for p in [',', '.', '!', '?']}
    for i, token in enumerate(toks.input_ids[0]):
        if token in punctuation_tokens:
            out[i, 0] = punctuation_weight
    # Ensure no row is all zeros
    for row in range(len_seq): 
        if out[row].sum() == 0:
            out[row, -1] = 1.0 / (len_seq - 1)
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize return 'Sentence Boundary and Punctuation Attention', out
    return "Sentence Boundary and Punctuation Attention", out