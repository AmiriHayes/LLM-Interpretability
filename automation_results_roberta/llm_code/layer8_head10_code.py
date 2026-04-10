import numpy as np
from transformers import PreTrainedTokenizerBase

def sentence_transition_focus(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Focusing on transitions: CLS, punctuations (like period, comma), and SEP tokens
    # Assume tokenizer converts these to token IDs: cls_id, sep_id, punct_ids
    cls_tok_id = toks.input_ids[0, 0]
    sep_tok_id = toks.input_ids[0, -1]
    punct_token_ids = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(p)) for p in ['.', ',', '!', '?', ':', ';']]
    flat_punct_ids = [item for sublist in punct_token_ids for item in sublist]  # Flatten list  

    # Modify matrix to focus on sentence transitions
    for i in range(len_seq):
        # Attention to CLS
        out[i, 0] = 1
        # Attention to SEP to illustrate end focus
        out[i, len_seq-1] = 1
        # Add noticeable weights for punctuations
        for j in range(1, len_seq - 1):
            if toks.input_ids[0, j] in flat_punct_ids:
                out[i, j] = 0.5

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize output to resemble attention patterns
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)
    return "Sentence Transition Focus Head", out