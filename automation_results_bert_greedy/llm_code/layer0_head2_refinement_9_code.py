import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase


def token_identity_entity_linking(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Token to position mapping, excluding special tokens
    spacy_tokens = sentence.split()
    token_alignment = {}
    j = 1  # Skip [CLS]
    for i, token in enumerate(toks.input_ids[0].numpy()):
        while j < len(spacy_tokens) and tokenizer.decode([toks.input_ids[0][i]]) in spacy_tokens[j]:
            token_alignment[j] = i
            j += 1

    # Implementing Token Identity and Entity Linking Pattern
    for entity_word_pair in [(transformed pair) for pair in sentence_attention.split() if int(pair.split(':')[1]) > threshold]:
        first_token, second_token = entity_word_pair.split('|')
        first_index = token_alignment.get(first_token)
        second_index = token_alignment.get(second_token)
        if first_index is not None and second_index is not None:
            out[first_index, first_index] = 1  # Self-link
            out[first_index, second_index] = 1
            out[second_index, first_index] = 1

    # Ensure that no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the matrix
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return 'Token Identity and Entity Linking', out