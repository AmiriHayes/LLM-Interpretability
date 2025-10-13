from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

def initial_token_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # The first non-special token after CLS often dominates attention
    main_tokens = [i for i in range(len_seq) if toks.input_ids[0][i] != tokenizer.cls_token_id and toks.input_ids[0][i] != tokenizer.sep_token_id]
    if main_tokens:
        first_non_special = main_tokens[0]
        out[first_non_special, :] = 1

    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0  # attend to EOS as a default

    # Return pattern name and predicted matrix
    return "Initial Token Dominance Attention", out