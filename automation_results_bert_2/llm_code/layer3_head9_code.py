import numpy as np
import spacy
from transformers import PreTrainedTokenizerBase

nlp = spacy.load('en_core_web_sm')


def noun_pronoun_association(sentence: str, tokenizer: PreTrainedTokenizerBase)
-> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    doc = nlp(sentence)

    # Create a mapping from spacy token index to transformer token index
    alignment = {}
    spacy_idx = 1  # Skip the first CLS token
    for i, word in enumerate(sentence.split()):
        alignment[i] = spacy_idx
        # Account for subwords in tokenizer
        subwords = tokenizer(word, add_special_tokens=False).input_ids
        spacy_idx += len(subwords)

    # Iterate over tokens and set attention matrix
    for spi, spacy_tok in enumerate(doc):
        if spacy_tok.pos_ in {"NOUN", "PROPN"}:
            for spj in range(spi+1, len(doc)):
                # Check for associating noun or pronoun with related objects
                if doc[spj].pos_ == "NOUN" or (doc[spj].pos_ == "PRON" and spacy_tok.i != doc[spj].head.i):
                    out[alignment[spi]+1, alignment[spj]+1] = 1
                    out[alignment[spj]+1, alignment[spi]+1] = 1

    # Ensure every token gets some attention
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4 # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize

    return "Noun-Noun and Pronoun-Object Memorable Association", out