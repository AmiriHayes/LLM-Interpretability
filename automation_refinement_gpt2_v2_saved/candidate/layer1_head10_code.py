import numpy as np
from transformers import PreTrainedTokenizerBase

def lexical_role_emphasis(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    special_tokens = [toks.input_ids[0][0].item(), toks.input_ids[0][-1].item()]
    for i in range(len_seq):
        if toks.input_ids[0][i].item() not in special_tokens:
            for j in range(len_seq):
                if toks.input_ids[0][j].item() == toks.input_ids[0][i].item():
                    out[i, j] = 1

    out[0, 0] = 1  # CLS token attention
    out[-1, 0] = 1  # EOS token attends to CLS

    out = (out + 1e-4) / (out.sum(axis=1, keepdims=True) + 1e-4)  # Normalize

    return "Lexical Role Emphasis Pattern", out

