import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

def action_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    verbs = set(["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"])
    tok_map = {}

    try:
        import spacy
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(sentence)
        sentence_map = [tok.i for tok in doc]
        for i, token in enumerate(toks.input_ids[0]):
            tok_map[token.item()] = sentence_map[i]
    except:
        print("SpaCy not available or model not loaded correctly.")
        return "Verb/Dominant Action Focus", out

    for i, token in enumerate(doc):
        if token.tag_ in verbs:
            for j in range(len_seq):
                out[i+1, j] = 1

    out = out / out.sum(axis=1, keepdims=True) # Normalize
    return "Verb/Dominant Action Focus", out