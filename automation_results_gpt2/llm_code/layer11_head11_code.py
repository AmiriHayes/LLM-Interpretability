import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase
import re

def subject_pronoun_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    words = sentence.split()

    # Define a list of common subject pronouns
    subject_pronouns = ['I', 'you', 'he', 'she', 'it', 'we', 'they', 'Lily', 'Her', 'His']

    # Loop through the tokenized sentence and identify subject pronouns
    for idx, word in enumerate(words):
        # Strip punctuations by checking only the first word part (useful with `token` output)
        word_clean = re.sub(r'[^\w]', '', word)

        # If the word is a subject pronoun, assign attention to it
        if word_clean in subject_pronouns:
            out[idx+1] *= 0  # Reset existing attention if any
            out[idx+1, idx+1] = 0.5  # Self-attention primarily to hold a certain pattern
            for jdx in range(len_seq):
                if jdx == idx + 1:
                    continue
                out[idx+1, jdx] = 0.5 / (len_seq - 1)  # Distribute attention across other tokens as fronts are already managed by self

    # Normalize the attention matrix
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0  # Ensure no row is all zeros
        else:
            out[row] = out[row] / out[row].sum()  # Normalize non-zero rows

    return "Subject Pronoun Attention", out