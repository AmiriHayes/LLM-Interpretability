from transformers import PreTrainedTokenizerBase
import numpy as np

def initial_token_specificity(sentence: str, tokenizer: PreTrainedTokenizerBase):
    # Tokenize the input sentence
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # The hypothesis is that the head attends specifically to the initial token
    # and propagates this high attention to other related tokens in its context.
    initial_token = toks.input_ids[0][1]  # Assuming first token is a special token

    # Assign the initial attention pattern
    for i in range(1, len_seq-1):
        out[i, 1] = 1.0  # direct attention to the first main token
        out[1, i] = 1.0

    # Set self-attention on residue non-first tokens
    for i in range(1, len_seq-1):
        out[i, i] = 0.5  # half attention on itself as a balance

    # Handle special tokens at the beginning and end
    out[0,0] = 1.0  # CLS/SOS token
    out[-1,0] = 1.0  # EOS token

    # Normalize the attention matrix by row
    out = out / out.sum(axis=1, keepdims=True)

    return "Initial Token-Specificity", out