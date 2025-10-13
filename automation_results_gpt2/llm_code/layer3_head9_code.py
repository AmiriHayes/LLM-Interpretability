import numpy as np
from transformers import PreTrainedTokenizerBase, GPT2TokenizerFast
from typing import Tuple


def subject_reference_resolution(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize the sentence into words
    words = sentence.split()
    # For ease of mapping token indices, assuming each word token belongs to respective indices
    token_index_mapping = {i: i+1 for i in range(len(words))}  # Plus one due to [CLS] token

    # Hypothesize subject tokens based on simplified NLP assumption (typically first noun phrase or pronoun)
    # Creating a simple mapping by checking usual sentence structures
    for i, word in enumerate(words):
        if word.lower() in {'i', 'he', 'she', 'they', 'we', 'lily', 'mom'}:
            subject_index = token_index_mapping[i]
            out[subject_index, subject_index] = 1  # Self-reference

            # A simplistic example of targeting likely verbs or related words
            linked_words = ['found', 'knew', 'wanted', 'went', 'said', 'asked', 'smiled', 'thanked', 'felt']
            for j, link_word in enumerate(words):
                if link_word.lower() in linked_words:
                    link_index = token_index_mapping[j]
                    out[subject_index, link_index] = 1
                    out[link_index, subject_index] = 1

    # Ensure no row in the matrix is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the matrix
    out = out / out.sum(axis=1, keepdims=True)

    return "Subject Reference Resolution", out