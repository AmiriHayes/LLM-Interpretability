import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load('en_core_web_sm')

def anaphoric_reference_resolution(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors='pt', truncation=True, max_length=512)
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize using spaCy for linguistic features
    doc = nlp(sentence)

    # Find all reference pairs using heuristic
    references = {}
    for index, token in enumerate(doc):
        if token.pos_ == 'PRON':
            # Refer back to the nearest named entity or significant noun
            for reverse_index in range(index - 1, -1, -1):
                reverse_token = doc[reverse_index]
                if reverse_token.ent_type_ or reverse_token.pos_ in ['NOUN', 'PROPN']:
                    references[index] = reverse_index
                    break

    # Map spaCy tokens to tokenizer tokens
    word_ids = toks.word_ids(batch_index=0)
    spaCy_to_tokenizer_index = {i: j for i, j in enumerate(word_ids) if j is not None}

    # Fill the attention matrix
    for pronoun_index, ref_index in references.items():
        if pronoun_index in spaCy_to_tokenizer_index and ref_index in spaCy_to_tokenizer_index:
            targ = spaCy_to_tokenizer_index[pronoun_index]
            src = spaCy_to_tokenizer_index[ref_index]
            out[targ, src] = 1
            out[src, targ] = 1

    # Normalize attention matrix
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0
    out = out / out.sum(axis=1, keepdims=True)

    return "Anaphoric Reference Resolution", out