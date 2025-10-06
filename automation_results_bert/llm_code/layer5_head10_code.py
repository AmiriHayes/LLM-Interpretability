import numpy as np
from transformers import PreTrainedTokenizerBase
import spacy

nlp = spacy.load("en_core_web_sm")

def conjunction_causality_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Process sentence with spaCy for dependency parsing
    doc = nlp(sentence)

    # Build alignment map between spaCy's tokens and the tokenizer's tokens
    alignment_map = {i: tok.i for i, tok in enumerate(doc) if i < len_seq-2}

    # Look for conjunctions and causality indicators, such as 'and', 'because', etc.
    conjunction_indices = {tok.i for tok in doc if tok.dep_ in ['cc', 'mark']}
    cause_effect_pairs = [(token.head.i, token.i) for token in doc if token.text.lower() in ['because', 'so']]

    # Apply activation for conjunctions and causal relations;
    for head_idx, dep_idx in cause_effect_pairs:
        if head_idx in alignment_map and dep_idx in alignment_map:
            spacy_head = alignment_map[head_idx]
            spacy_dep = alignment_map[dep_idx]
            out[spacy_head+1, spacy_dep+1] = 1
            out[spacy_dep+1, spacy_head+1] = 1

    for conj_idx in conjunction_indices:
        if conj_idx in alignment_map:
            spacy_conj = alignment_map[conj_idx]
            out[spacy_conj+1] = 1

    # Normalize out matrix by row
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0
        out[row] += 1e-4
        out[row] = out[row] / out[row].sum()

    return "Conjunction and Causality Relationships", out