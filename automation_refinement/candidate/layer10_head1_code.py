import numpy as np
from transformers import PreTrainedTokenizerBase

# Function to predict clausal boundaries in a sentence

def clausal_boundary_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # Loop through each token in the sentence
    for i in range(1, len_seq - 1):
        # Check tokens that are punctuation marks likely indicating boundaries
        if sentence[toks.token_to_chars(i)[0]:toks.token_to_chars(i)[1]] in {',', '.', ';', '?', '!', ':', '-'}:
            # Set attention pattern typical of boundary focus
            out[i, i] = 1
            # Additive attention to adjacent tokens representing boundary role
            if i + 1 < len_seq:
                out[i, i + 1] = 0.5
            if i - 1 > 0:
                out[i, i - 1] = 0.5
            out[0, 0] = 1
            out[-1, 0] = 1
    # Normalize out matrix by row (results in uniform attention)
    out += 1e-4
    out /= out.sum(axis=1, keepdims=True)
    return "Clausal and Sub-clausal Boundary Attention", out