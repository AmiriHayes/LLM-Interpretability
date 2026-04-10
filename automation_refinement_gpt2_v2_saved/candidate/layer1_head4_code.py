import numpy as np
from transformers import PreTrainedTokenizerBase

def function_def_header_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    def_id_in_attention = False
    for i in range(len_seq - 1):
        # Identify the definition beginning by searching common keywords
        if any(tokenizer.decode(toks.input_ids[0][i]).strip() == kw for kw in ['def', 'class', 'function']):
            # Give attention to the definition header
            def_id_in_attention = True
            out[i, :i+1] = 1

    if def_id_in_attention:
        # Normalize the attention weights
        out[0, 0] = 1
        out[-1, 0] = 1
        out += 1e-4
        out = out / out.sum(axis=1, keepdims=True)

    return "Function Definition Header Attention", out