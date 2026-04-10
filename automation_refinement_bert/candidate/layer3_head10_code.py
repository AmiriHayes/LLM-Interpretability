import numpy as np
from transformers import PreTrainedTokenizerBase

def compound_modifier_patterns(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    words = sentence.split()
    idx_mapping = {i: tok_id for i, tok_id in enumerate(toks.word_ids()[0])}

    compounds = []
    # Here we identify compound indexes from attention (mock logic since we don't have real dependency parsing)
    for i, tok_pair in enumerate(sentence.split()):
        if "##" in tok_pair:  # very simplistic detection of sub-word
            compound_word_idx = [k for k, v in idx_mapping.items() if v == i]
            if compound_word_idx:
                compounds.append(compound_word_idx)

    for comp in compounds:
        for i in range(len(comp) - 1):
            out[comp[i], comp[i + 1]] = 1.0
            out[comp[i + 1], comp[i]] = 1.0

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize return matrix

    return "Compound Modifier Patterns", out