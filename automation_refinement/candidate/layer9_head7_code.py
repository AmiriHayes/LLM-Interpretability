import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase
import spacy

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

def complex_sentence_structure(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Analyze the sentence using spaCy
    doc = nlp(sentence)
    token_to_idx = {}
    token_counter = 1

    for token in doc:
        token_text = token.text
        if token_text in toks.tokenizer:
            token_to_idx[token_text] = token_counter
            token_counter += 1

    # Capture structure within complex or multi-part elements
    for tok in doc:
        if len(tok.text) > 2 and tok.pos_ in {"NOUN", "ADJ", "ADV"}:
            head_index = token_to_idx.get(tok.text.lower(), 0)
            for child in tok.children:
                child_index = token_to_idx.get(child.text.lower(), 0)
                out[head_index, child_index] = 1

    # Establish some attention to the separators and sentence boundaries
    out[0, 0] = 1
    out[-1, 0] = 1
    out = out / out.sum(axis=1, keepdims=True)
    return "Complex Sentence Structure Association", out

