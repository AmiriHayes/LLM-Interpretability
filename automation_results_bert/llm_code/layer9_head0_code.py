import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase
import spacy

# Load spaCy's English tokenizer
nlp = spacy.load('en_core_web_sm')

def object_subject_relationships(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    doc = nlp(sentence)
    token_map = {token.idx: i+1 for i, token in enumerate(doc) if i+1 < len_seq}  # Token index to matrix index mapping

    for token in doc:
        # Focus on subject and object dependencies
        if "obj" in token.dep_ or "subj" in token.dep_:
            head_index = token.head.i + 1  # Head's position in out matrix
            token_index = token_map.get(token.idx, None)
            if token_index is not None:
                out[head_index, token_index] = 1.0
                out[token_index, head_index] = 1.0

    # Ensure CLS and SEP attend to themselves
    out[0, 0] = 1.0
    out[-1, -1] = 1.0

    # Ensure no row is entirely zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Object and Subject Relationships", out