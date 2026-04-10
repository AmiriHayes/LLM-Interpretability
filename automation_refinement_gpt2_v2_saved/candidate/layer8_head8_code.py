from transformers import PreTrainedTokenizerBase
import numpy as np
from typing import Tuple

# Hypothesis: Layer 8, Head 8 focuses specifically on the line start tokens, conveying the role of managing line-by-line attention in a code document.
def line_focus_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # Assumption: Line starts are marked by newline tokens
    line_starts = [i for i, tok_id in enumerate(toks.input_ids[0]) if toks.input_ids[0][i - 1] == tokenizer.encode('\n')[0]]
    for i in line_starts:
        for j in range(len_seq):
            out[j, i] = 1
    # Ensure every token attends to itself
    for i in range(len_seq):
        out[i, i] = 1
    # Normalize the matrix
    out = out / out.sum(axis=1, keepdims=True)
    return "Line Focus Attention", out