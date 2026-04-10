import numpy as np
from transformers import PreTrainedTokenizerBase


def func_definition_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    words = sentence.split()
    # Let's find the index of 'def' as it seems to focus heavily in the examples
    def_index = None
    for idx, word in enumerate(words):
        if 'def' in word:
            def_index = idx
            break

    if def_index is not None:
        # Ensuring attention is on the "def" and potentially the function definition
        for idx in range(len_seq):
            if idx == def_index + 1:
                out[idx, idx] = 1
            elif idx < def_index + 1:
                # emphasize tokens before for function header understanding
                out[idx, def_index + 1] = 0.5
            else:
                # Later tokens may have reduced emphasis
                out[idx, def_index + 1] = 0.2

    out[0, 0] = 1
    out[-1, 0] = 1
    out = out / np.sum(out, axis=1, keepdims=True)  # Normalize
    return "Function Definition Focusing", out