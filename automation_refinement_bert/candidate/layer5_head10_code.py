import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple
import spacy

nlp = spacy.load('en_core_web_sm')

def comma_pairing_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize with spaCy for potential word-to-token alignment
    doc = nlp(sentence)
    token_to_word_map = {i: token.i for i, token in enumerate(doc)}

    # Iterate through tokens to identify comma pairs (sequences between commas)
    comma_indices = [i for i, token in enumerate(doc) if token.text == ',']

    # Iterate over detected comma pairs
    for start, end in zip(comma_indices, comma_indices[1:]):
        tokens_in_between = list(range(start + 1, end))
        # Ensure no comma has zero attention
        for i in tokens_in_between:
            for j in tokens_in_between:
                out[token_to_word_map[i], token_to_word_map[j]] = 1

    # Normalize the attention weights
    out += 1e-4 # Avoid division by zero
    out /= out.sum(axis=1, keepdims=True)

    # Ensure no attention rows are entirely zero
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return 'Comma Pairing Attention', out