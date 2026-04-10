from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

def sentence_boundary_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Focus CLS [0, 0] and SEP [end, end]
    out[0, 0] = 1.0
    out[-1, -1] = 1.0

    # Iterate over the tokens to apply punctuation and boundary focus
    for i, token_id in enumerate(toks.input_ids[0]):
        token_str = tokenizer.convert_ids_to_tokens(token_id.item())

        # Check if the token is recognized as punctuation
        if token_str in {".", ",", "!", "?", "</s>", "<s>"}:
            # Assign high attention weight to the CLS
            out[i, 0] = 1.0
            # Pay attention to the SEP
            out[i, -1] = 0.5

    # Normalize the matrix so each row sums to 1
    out += 1e-4  # avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return "Punctuation and Sentence Boundary Focus", out