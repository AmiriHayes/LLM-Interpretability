from transformers import PreTrainedTokenizerBase
import numpy as np
import spacy
from typing import Tuple

nlp = spacy.load('en_core_web_sm')

def subject_verb_agreement(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Parse sentence with spaCy to get dependencies
    doc = nlp(sentence)
    tok_map = {}
    for i, tok_id in enumerate(toks.word_ids()):
        tok_map[i] = tok_id

    for token in doc:
        # Check if the token is a subject (nsubj) or verb (VERB)
        if token.dep_ == 'nsubj' and token.head.pos_ == 'VERB':
            subj_index = next((i for i, wid in tok_map.items() if wid == token.i), None)
            verb_index = next((i for i, wid in tok_map.items() if wid == token.head.i), None)
            if subj_index is not None and verb_index is not None:
                # Indicate strong attention between subject and its verb
                out[subj_index, verb_index] = 1.0
                out[verb_index, subj_index] = 1.0

    # If no attention was found, mark attention to the last token for normalization purposes
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize attention scores
    return "Subject-Verb Agreement Emphasis", out