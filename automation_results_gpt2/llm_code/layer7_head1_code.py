from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load('en_core_web_sm')

def coreference_resolution(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Use spaCy to parse the sentence
    doc = nlp(sentence)
    token_to_index = {tok.idx: i+1 for i, tok in enumerate(doc) if tok.text.strip()}

    # Iterate over tokens and activate coreference patterns
    for i, token in enumerate(doc):
        if token.pos_ == "PRON" or token.dep_ == "nsubj":
            # Find previous tokens possibly acting as antecedents
            for j in range(i):
                if doc[j].pos_ in ["NOUN", "PROPN"]:
                    if j in token_to_index and i in token_to_index:
                        out[token_to_index[j], token_to_index[i]] = 1
                        out[token_to_index[i], token_to_index[j]] = 1

    # Assign self attention for cls ([0, 0]) and eos ([-1, -1]), and default attention to [eos, ...]
    out[0, 0] = 1
    out[-1, -1] = 1
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Coreference Resolution Pattern", out