from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase


def function_def_unrolling(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Extracting tokenized words and establishing a mapping
    token_to_id = {token_id: idx for idx, token_id in enumerate(toks.input_ids[0].tolist())}
    words = tokenizer.convert_ids_to_tokens(toks.input_ids[0], skip_special_tokens=False)

    # Assume function structure starts with 'def', has colon ':' and possible '
)' for parameters
    if 'def' in words:
        def_idx = words.index('def')
        colon_idx = words.index(':') if ':' in words else -1
    else:
        def_idx, colon_idx = -1, -1

    # Mark attention from 'def' to start of function parameters (until ':' or first '(')
    for i in range(def_idx + 1, min(colon_idx, len_seq - 1)):
        out[def_idx, i] = 1

    # Conclusion sentence tail point to function head
    if colon_idx != -1:
        for i in range(colon_idx + 1, len_seq - 1):
            if words[i] == '\n':  # denotes end of statement
                out[def_idx, i] = 1
                break

    # Standard procedure for CLS and output initialization
    out[0, 0] = 1
    out[-1, 0] = 1

    return "Function Definition and Unrolling Pattern", out

