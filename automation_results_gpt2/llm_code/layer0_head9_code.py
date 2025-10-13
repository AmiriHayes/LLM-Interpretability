import numpy as np
from transformers import PreTrainedTokenizerBase

def first_token_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # The first real token after CLS (index 1) focuses on itself significantly
    main_focus_index = 1
    for i in range(1, len_seq - 1):
        out[main_focus_index, i] = 1

    # Make sure every token focuses a little on itself and to CLS
    np.fill_diagonal(out, 1)

    # Normalize each row to ensure it's distributed properly
    norm_factor = out.sum(axis=1, keepdims=True)
    norm_factor[norm_factor == 0] = 1  # Prevent division by zero
    out = out / norm_factor

    return "First Token Focus", out