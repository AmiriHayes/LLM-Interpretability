import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

def conjunctive_coordination_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Map token indices to their original sentence words
    spacy_to_transformers_token_mapping = {}
    for i, word_idx in enumerate(toks.word_ids()):
        if word_idx is not None:
            spacy_to_transformers_token_mapping.setdefault(word_idx, []).append(i)

    # Tokenized word list
    words = sentence.split()

    # Rule: if a conjunction is found, attention is on both parts
    for i, word in enumerate(words):
        if word.lower() in ['and', 'or', 'but']: # Common conjunctions
            if i > 0:
                prev_tokens = spacy_to_transformers_token_mapping.get(i-1, [])
                conj_tokens = spacy_to_transformers_token_mapping.get(i, [])
                next_tokens = spacy_to_transformers_token_mapping.get(i+1, [])

                # Set attention between conjunction and its surrounding parts
                for p_token in prev_tokens:
                    for c_token in conj_tokens:
                        out[c_token, p_token] = 1

                for n_token in next_tokens:
                    for c_token in conj_tokens:
                        out[c_token, n_token] = 1

    # Ensure there's always attention to the sentence end
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4 # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True) # Normalize

    return "Conjunctive Coordination Focus", out