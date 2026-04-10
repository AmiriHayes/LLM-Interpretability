import numpy as np
from transformers import PreTrainedTokenizerBase


def def_token_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Identify the index of the 'def' token
    def_index = -1
    for i, token_id in enumerate(toks.input_ids[0]):
        if tokenizer.convert_ids_to_tokens([token_id])[0] == 'Ġdef':
            def_index = i
            break

    if def_index != -1:
        # If 'def' is found, focus the entire sentence attention on it
        out[:, def_index] = 1
        out[0, 0] = 1    # CLS self-attention
        out[-1, 0] = 1   # EOS self-attention

    return "Def Token Focus", out