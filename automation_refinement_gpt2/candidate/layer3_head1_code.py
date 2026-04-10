import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load('en_core_web_sm')

def initial_and_clause_emphasis(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    words = [t.text for t in nlp(sentence)]
    tok_id_map = {i: token_idx for token_idx, i in enumerate(toks.word_ids()) if i is not None}

    # Clause detection via punctuation
    clause_boundary_indices = [i for i, tok in enumerate(words) if tok in {',', ';', ':', '?', '!', '.'}]
    if clause_boundary_indices:
        for end_idx in clause_boundary_indices:
            for start_idx in range(end_idx):
                if start_idx in tok_id_map and end_idx in tok_id_map:
                    out[tok_id_map[start_idx]+1, tok_id_map[end_idx]+1] = 1

    # Emphasis on initial token
    out[1, :] = 1/len_seq

    for row in range(len_seq):  # Ensure no row is all zeros
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Initial Token and Subsequent Clause Emphasis", out