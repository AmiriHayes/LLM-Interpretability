import numpy as np
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing import Tuple

def subject_pronoun_initial_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    '''
    This function predicts an attention pattern where the head primarily focuses on the initial subject pronoun 
    or the first word of the sentence, spreading its attention from this point throughout the sequence.
    '''
    # Tokenize the input sentence
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Identify first significant token (usually the subject or initial main token)
    # Default to the first token (after any potential special tokens)
    first_sig_idx = 1 

    # Give full attention to the first significant token
    out[first_sig_idx, first_sig_idx] = 1.0

    # Simulate attention spreading from the first significant token to other tokens
    # Optional attention dispersion (simplification)
    for i in range(1, len_seq - 1):
        if i != first_sig_idx:
            out[first_sig_idx, i] = 1.0 / (len_seq - 2)

    for i in range(len_seq):
        # Avoid rows with no assigned attention to keep uniformity
        if out[i].sum() == 0:
            out[i, -1] = 1.0

    # Normalize the attention output
    out = out / out.sum(axis=1, keepdims=True)

    return "Subject Pronoun Initial Attention", out