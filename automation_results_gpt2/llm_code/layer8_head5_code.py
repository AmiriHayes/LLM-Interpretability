import numpy as np
from transformers import PreTrainedTokenizerBase

def sentence_start_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # The assumption is the head attends primarily to the first token of the sentence.
    for i in range(1, len_seq-1):
        out[i, 1] = 1  # Attend to the start of the sentence

    # Ensure no token is not attended by at least adding a small value to the last token
    out[:, -1] = 1e-4  # Little attention to end token to ensure normalization doesn't break.

    # Normalize each row to sum to 1
    out /= out.sum(axis=1, keepdims=True)

    return "Sentence Start Attention Pattern", out