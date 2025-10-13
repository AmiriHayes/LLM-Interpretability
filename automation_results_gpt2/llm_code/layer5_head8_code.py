import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

def subject_attention_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    words = sentence.split()

    # Determine the index of the first subject-like word
    subject_indices = range(1, len_seq)  # All words starting from the first token

    # Loop through token indices
    for i in subject_indices:
        # Assign significant attention to the 'subject' of the sentence
        out[i, 0] = 1  # All tokens attend to their subject

    # Assign default attentions to [CLS] and [SEP] tokens
    out[0, 0] = 1
    out[-1, 0] = 1

    # Normalize attention scores
    out = out + 1e-4 # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True) # Normalize 

    return "Subject Attention Pattern", out