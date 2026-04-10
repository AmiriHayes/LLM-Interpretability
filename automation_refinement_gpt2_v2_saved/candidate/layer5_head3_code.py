import numpy as np
from transformers import PreTrainedTokenizerBase

def function_definition_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    token_indices = {i: tok for i, tok in enumerate(toks.input_ids[0])}

    # Attention boosting for 'def' token
    def_indices = [i for i, tok in token_indices.items() if tokenizer.decode([tok]) == 'def']

    for def_index in def_indices:
        for i in range(len_seq):
            out[i, def_index] = 1.0

    # Self-attention patterns for first and last tokens (normally CLS and EOS)
    out[0, 0] = 1
    out[-1, 0] = 1

    # Normalize output
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    return "Function Definition Attention Pattern", out