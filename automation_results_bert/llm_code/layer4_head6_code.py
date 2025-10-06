import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load('en_core_web_sm')

def subordinate_clause_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    words = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    word_id_map = {i: word for i, word in enumerate(words)}
    doc = nlp(sentence)

    subordinate_clauses = {"because", "so", "if", "although", "that", "which"}
    clause_head_indices = [i for i, token in enumerate(doc) if token.text.lower() in subordinate_clauses]

    for i, token in enumerate(doc):
        # Attention from subordinate clause marker to its governing verb/noun
        if i in clause_head_indices:
            for ancestor in token.ancestors:
                out[token.i+1, ancestor.i+1] = 1

        # Attention from governing verb/noun to subordinate clause marker
        for child in token.children:
            if child.i in clause_head_indices:
                out[token.i+1, child.i+1] = 1

    # CLS and SEP token self-attention
    np.fill_diagonal(out, 1)

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize
    return "Subordinate Clause Attention Pattern", out