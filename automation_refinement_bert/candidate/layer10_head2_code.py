import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load('en_core_web_sm')

def dependency_parsing(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    words = sentence.split()
    doc = nlp(sentence)
    spaCy_to_bert = {i: j + 1 for i, j in enumerate(doc)
                   if tokenizer.convert_tokens_to_ids(j.orth_) != toks.input_ids[0, i + 1].item()}
    for stok in doc:
        parent_index = stok.i
        if parent_index in spaCy_to_bert:
            parent_index = spaCy_to_bert[parent_index]
        for child_stok in stok.children:
            child_index = child_stok.i
            if child_index in spaCy_to_bert:
                child_index = spaCy_to_bert[child_index]
            out[parent_index, child_index] = 1
            out[child_index, parent_index] = 1
    out[0, 0] = 1
    out[-1, 0] = 1
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)
    return "Dependency Parsing Pattern", out