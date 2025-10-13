import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

def subject_pronoun_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Hypothesis: This head focuses on the subject pronoun heavily and assigns some distribution 
    # to the first verb following it, mimicking attention on the subject and its action.

    # Tokenizes the sentence into words
    words = sentence.split()
    subject_indices = []  # indices in the tokenized version

    # Basic identification of subject pronouns through simple patterns:
    subject_pronouns = ["I", "you", "he", "she", "it", "we", "they", "She", "He", "They"]

    # Identify the position of the subject pronouns in the tokenized sentence
    for i, word in enumerate(words):
        if any(sp in word for sp in subject_pronouns):
            try:
                token_index = toks.word_ids().index(i)
                subject_indices.append(token_index)
                # Giving attention heavily to itself (subject pronoun itself)
                out[token_index, token_index] = 1.0

                # Finding the first verb or action after the pronoun
                if token_index < len_seq - 1:
                    out[token_index, token_index + 1] = 0.5  # Give half attention to next token assuming it's an action
            except ValueError:
                continue

    # Ensure no row is all zeros (uniform distribution as fallback)
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalization of the attention weights
    out += 1e-4  # To avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)
    return "Subject Pronoun Focus Head", out