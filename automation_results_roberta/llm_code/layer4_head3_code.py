from transformers import PreTrainedTokenizerBase
import numpy as np

def cls_specialization(sentence: str, tokenizer: PreTrainedTokenizerBase) -> str:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # The <s> token is often the most attended-to token
    out[:, 0] = 1  # All tokens attend to the <s> token

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize for valid attention distribution
    out = out / out.sum(axis=1, keepdims=True)

    return "CLS-Specialization", out