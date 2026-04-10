import numpy as np
from transformers import PreTrainedTokenizerBase


def sentence_start_attention_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assuming the model would learn to focus primarily on the start token <s>
    # For this head, all tokens align with <s> according to the pattern observed.
    for i in range(1, len_seq-1):
        out[i, 0] = 1.0

    # Ensure end of sentence token receives some attention
    out[0, 0] = 1.0
    out[-1, 0] = 1.0

    # Normalize each row to avoid any zero rows
    out += 1e-4  # Tiny constant to avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return "Sentence Start Attention Pattern", out