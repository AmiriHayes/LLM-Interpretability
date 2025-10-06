from transformers import PreTrainedTokenizerBase
from typing import Tuple
import numpy as np

def complement_verb_association(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    word_to_token_map = {}
    offset = 0
    for idx, input_id in enumerate(toks.input_ids[0]):
        token = tokenizer.decode([input_id])
        word_to_token_map[offset] = idx
        if token.strip() == '' or token in ['[CLS]', '[SEP]']:
            continue
        offset += 1

    spacy_words = sentence.split()

    for i, spacy_word in enumerate(spacy_words[:-1]):
        if spacy_word in ['to', 'with', 'because', 'and', 'for', 'them']:
            verb_index = i + 1
            if verb_index < len(spacy_words):
                token_index = word_to_token_map.get(i, None)
                verb_token_index = word_to_token_map.get(verb_index, None)
                if token_index is not None and verb_token_index is not None:
                    out[token_index, verb_token_index] = 1

    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out /= out.sum(axis=1, keepdims=True)
    return "Complement Verb Association", out