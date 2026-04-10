from typing import Tuple
import numpy as np
import spacy
from transformers import PreTrainedTokenizerBase

nlp = spacy.load('en_core_web_sm')

def verb_contextual_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    doc = nlp(sentence)
    verb_indices = {token.i: idx + 1 for idx, token in enumerate(doc) if token.pos_ == 'VERB'}

    for token in doc:
        token_index = token.i + 1
        if token.i in verb_indices:
            # Strong attention from verb to its context (nearby tokens)
            start = max(1, token_index - 3)
            end = min(len_seq - 1, token_index + 3)
            for i in range(start, end + 1):
                out[token_index, i] = 1
                out[i, token_index] = 1

        closest_verb_index = min(verb_indices.keys(), key=lambda vi: abs(vi - token.i), default=None)
        if closest_verb_index is not None:
            closest_verb_index += 1
            out[token_index, closest_verb_index] = 1
            out[closest_verb_index, token_index] = 1

    # Ensure no row is entirely zeros by letting every token attend to [SEP]
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, len_seq - 1] = 1.0

    # Normalize the output matrix
    out += 1e-4
    out /= out.sum(axis=1, keepdims=True)

    return "Verb-Centered Contextual Focus", out