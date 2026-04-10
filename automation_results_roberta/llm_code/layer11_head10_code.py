import numpy as np
from transformers import PreTrainedTokenizerBase

# Hypothesis: Sentence Start Anchoring - This head predominantly attends to the start of the sentence token ('<s>') across various sentences, regardless of the other contents in the sentence.

def sentence_start_anchoring(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # The pattern consistently shows attention to the <s> token for all non-special tokens
    for i in range(1, len_seq-1):  # Exclude <s> (position 0) and </s> (last position)
        out[i, 0] = 1  # Attention to <s>

    # Ensure the special tokens have some attention distribution
    out[0, 0] = 1  # <s> attends to itself
    out[-1, -1] = 1  # </s> attends to itself

    # Normalize attention matrix such that rows sum to 1, including adding a small constant for stability
    for row in range(len_seq):
        if out[row].sum() > 0:
            out[row] /= out[row].sum()
        else:
            out[row, -1] = 1  # Default to send ignored tokens to </s>

    return "Sentence Start Anchoring", out