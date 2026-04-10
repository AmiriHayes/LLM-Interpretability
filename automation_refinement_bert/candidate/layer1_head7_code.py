import numpy as np
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

def end_punctuation_boundary_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])

    # Identify the position of end punctuations such as '.', '?', and '!' in the sentence.
    end_punctuations = {'.', '?', '!'}  
    for i, token in enumerate(tokens):
        if any(punc in token for punc in end_punctuations):
            out[i, :] = 1.0
            out[:, i] = 1.0

    # Ensure no row is all zeros (assigns some residual attention to the [SEP] token)
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalizing the attention matrix
    out += 1e-4  # Avoid division by zero in any column
    out = out / out.sum(axis=1, keepdims=True)  # Normalize so each row sums to 1

    return "End Punctuation-Boundary Attention", out