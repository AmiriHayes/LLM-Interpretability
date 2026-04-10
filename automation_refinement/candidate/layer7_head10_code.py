from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load('en_core_web_sm')

def nominal_association(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize the sentence with spaCy for noun detection
    doc = nlp(sentence)
    noun_indices = [tok.i for tok in doc if tok.pos_ == 'NOUN']

    # Align token indices between spaCy and the tokenizer
    token_mapping = {w: i for i, w in enumerate(sentence.split())}

    # Iterate over the indices of nouns from spaCy in the sequence
    for noun_index in noun_indices:
        if noun_index in token_mapping:
            mapped_index = token_mapping[noun_index]
            # Give high attention to nouns that are far from each other
            for other_noun_index in noun_indices:
                if other_noun_index != noun_index and other_noun_index in token_mapping:
                    other_mapped_index = token_mapping[other_noun_index]
                    out[mapped_index+1, other_mapped_index+1] = 1

    # CLS and SEP token patterning
    out[0, 0] = 1
    out[-1, 0] = 1

    # Normalize
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    return "Nominal Association Pattern", out