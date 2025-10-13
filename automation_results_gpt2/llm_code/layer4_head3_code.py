import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

# This function predicts the attention pattern for Layer 4, Head 3 in GPT-2.
# It assumes the pattern observed is to focus on the sentence subject or pronoun.
def pronoun_reference(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # The focus is often on pronouns or the subject of the sentence, represented by the first non-special token
    # We assume it projects to the corresponding tokens with decreasing strength from the first pronoun/subject
    # Typically, pronouns/subjects appear early in English sentences, usually near the beginning
    for i in range(1, len_seq):  # Start from 1 to avoid [CLS] token
        out[i][1] = 1.0  # Assuming the subject or pronoun from position 1 gets major attention

    for row in range(len_seq): # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the matrix so attention scores sum to 1
    out = out / out.sum(axis=1, keepdims=True)
    return 'Pronoun/Subject Reference Pattern', out