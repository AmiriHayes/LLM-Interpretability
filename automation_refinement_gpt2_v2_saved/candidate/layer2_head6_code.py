import numpy as np
from transformers import PreTrainedTokenizerBase


def function_definition_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    tokens = toks.tokens()

    # Track def and first function token
    def_indices = []
    first_token_after_def = None

    for i, token in enumerate(tokens):
        if "def" in token:
            def_indices.append(i)
        elif i > 0 and tokens[i-1] in def_indices and first_token_after_def is None:
            first_token_after_def = i

    if def_indices:
        first_token_after_def = first_token_after_def or (def_indices[-1] + 1)

        for i in range(len_seq):
            for def_idx in def_indices:
                out[i, def_idx] = 1
            out[i, first_token_after_def] = 1

    # Ensures CLS-like attention to the first and last token
    out[0, 0] = 1
    out[-1, 0] = 1

    # Normalize attention rows
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    return "Function Definition Focus", out