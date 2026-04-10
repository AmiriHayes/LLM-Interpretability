import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

def sentence_boundaries_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    # Tokenize the sentence and prepare the attention matrix
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Assign high attention to [CLS] and [SEP] tokens, which are the start and end markers
    for i in range(len_seq):
        out[i, 0] = 1  # Attention to <s>
        out[i, -1] = 1  # Attention to </s>

    # Ensure each row sums to avoid zero-sum problems
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0
        else:
            out[row, :] /= out[row, :].sum()  # Normalize the row

    return "Sentence Begin and End Focus", out