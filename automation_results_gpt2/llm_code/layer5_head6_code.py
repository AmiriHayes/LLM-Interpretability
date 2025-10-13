from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

def sentence_start_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    # Tokenize the input sentence
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Focusing attention mainly to the first non-[CLS] token
    primary_focus_index = 1  # Assuming the first token is [CLS]

    # Assign high attention to the tokens later in relation to the start token
    for i in range(1, len_seq):
        out[i, primary_focus_index] = 1
        out[primary_focus_index, i] = 1

    # Ensure CLS ([CLS]) and SEP ([SEP]) tokens are attended to correctly
    out[0, 0] = 1  # CLS token self-attention
    out[-1, -1] = 1  # SEP token self-attention

    # Normalize attention matrix
    out = out / out.sum(axis=1, keepdims=True)
    return "Sentence Start Focus Pattern", out