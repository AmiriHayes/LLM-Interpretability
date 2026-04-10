import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

def function_declaration_alignment(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])

    function_start_idx = None

    # Identify the start index of each function-like declaration.
    for idx, token in enumerate(tokens):
        if token == 'def' or token == 'class' or token.startswith('def') or (token.startswith('Ġ') and token.endswith(':')):
            function_start_idx = idx
            break

    # If a function declaration is identified, set attention from the function keyword to subsequent tokens.
    if function_start_idx is not None:
        for i in range(function_start_idx, len_seq):
            out[function_start_idx, i] = 1

    # Self-attention for the [CLS] token and the next token
    out[0, 0] = 1
    out[function_start_idx, 0] = 1

    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    return "Function Declaration Alignment", out