import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

def sentence_boundary_modifier_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Set self-attention on [CLS]/<s> token
    out[0, 0] = 1.0
    # Set attention to focus on sentence boundaries (e.g., [CLS]/<s> and [SEP]/</s>)
    for i in range(1, len_seq - 1):
        out[i, 0] = 0.5  # Pay some attention to <s>
        out[i, len_seq - 1] = 0.5  # Pay some attention to </s>

    # Normalize the attention matrix so that each row sums to 1 (except for [CLS])
    for i in range(1, len_seq - 1):
        attention_sum = out[i].sum()
        if attention_sum > 0:
            out[i] /= attention_sum

    # Ensure no attention rows are all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Sentence Boundary and Modifier Focus", out