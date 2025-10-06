from typing import Tuple
from transformers import PreTrainedTokenizerBase
import numpy as np
import spacy

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

def verb_complement_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Process sentence with spaCy
    doc = nlp(sentence)

    # Create mapping of token positions between spaCy and tokenizer
    token_mapping = {}
    token_idx = 1
    for token in doc:
        word_tokens = tokenizer.tokenize(token.text)
        for _ in word_tokens:
            token_mapping[token_idx] = token.i
            token_idx += 1

    # Assign attention based on verbs and complements/objects
    for token in doc:
        if token.pos_ == "VERB":
            # Find objects/complements of the verb
            for child in token.children:
                if child.dep_ in {"dobj", "pobj", "ccomp", "xcomp"}:
                    verb_idx = token.i
                    complement_idx = child.i
                    if verb_idx in token_mapping.values() and complement_idx in token_mapping.values():
                        verb_token_ids = [k for k, v in token_mapping.items() if v == verb_idx]
                        comp_token_ids = [k for k, v in token_mapping.items() if v == complement_idx]
                        for v_id in verb_token_ids:
                            for c_id in comp_token_ids:
                                out[v_id, c_id] = 1
                                out[c_id, v_id] = 1

    # Ensure CLS ([0, 0]) and SEP ([-1, 0]) attention
    out[0, 0] = 1
    out[-1, 0] = 1

    # Normalize the attention matrix to ensure no division by zero
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    return "Verbs and Their Complements/Objects Attention Pattern", out