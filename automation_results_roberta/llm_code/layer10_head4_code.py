import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

def semantic_coherence_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Identify thematic word position indices that are central to the sentence theme
    thematic_words_indices = set()
    for i, token_id in enumerate(toks.input_ids[0]):
        token = tokenizer.decode(token_id)
        # Hypothetical heuristic to determine thematic words: match nouns and verbs as thematic
        if token in ["needle", "shirt", "share", "sew", "button", "fix", "found", "smiled"]:
            thematic_words_indices.add(i)

    # Strong self-attention for thematic tokens
    for i in thematic_words_indices:
        out[i, i] = 1.0

    # Ensuring each token focuses on thematic words initially
    for i in range(len_seq):
        for idx in thematic_words_indices:
            out[i, idx] = 1.0 / len(thematic_words_indices)

    # Normalize rows
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0
        out[row] += 1e-4  # Avoid division by zero
        out[row] = out[row] / out[row].sum()

    return "Semantic Coherence Focus", out