import numpy as np
from transformers import PreTrainedTokenizerBase

def conjunction_and_causation_link(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    words = tokenizer.convert_ids_to_tokens(toks.input_ids[0])

    # Simple heuristic to handle conjunctions and causations
    for i, word in enumerate(words):
        if word in ['and', 'because', ',', 'but', 'so', 'so', 'or']:
            # Link the conjunction or causation marker to adjacent tokens
            if i > 0:
                out[i, i-1] = 1
                out[i-1, i] = 1
            if i < len_seq - 1:
                out[i, i+1] = 1
                out[i+1, i] = 1

    # Ensure each token has at least some minimal attention
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Conjunction and Causation Linking", out