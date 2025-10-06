import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple


def punctuation_and_boundary_awareness(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Identifying where punctuation marks occur in the sentence token ids
    punct_tokens = [token_id for token_id, token in enumerate(toks.input_ids[0]) if token in [1012, 102, 999]]  # . [SEP] ?

    # Assign attention based on punctuation for boundaries
    for tok_idx in range(1, len_seq - 1):
        if tok_idx in punct_tokens:
            # Attend to punctuation itself with high weight indicating boundary awareness
            for punct in punct_tokens:
                out[tok_idx, punct] = 1.0
        else:
            # Default to some attention to sentence boundaries otherwise
            out[tok_idx, -1] = 0.5  # [SEP]
            out[tok_idx, 0] = 0.5   # [CLS]

    # Normalize the attention matrix row-wise
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0
        out[row] /= out[row].sum()

    return "Punctuation and Boundary Awareness", out