import numpy as np
from transformers import PreTrainedTokenizerBase


def sentence_boundary_attention(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Focus the attention pattern on the first token (e.g., <s>) and the last token (e.g., </s>)
    for i in range(1, len_seq - 1):
        out[i, 0] = 1  # High attention on the initial token <s>
        out[i, -1] = 1  # High attention on the final token </s>

    # Ensure no token row is left with zero attention (meaningfully assigning some attention)
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the attention matrix row-wise
    out += 1e-4  # Avoid complete zeros in any row
    out = out / out.sum(axis=1, keepdims=True)

    return "Sentence Boundary Focusing", out