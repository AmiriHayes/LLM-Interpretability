import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

# The hypothesis suggests that this head focuses on coordinating conjunctions and their relationships in the sentence.
def cc_attention_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Get the token IDs and words for mapping
    token_ids = toks.input_ids[0].tolist()
    words = tokenizer.convert_ids_to_tokens(token_ids)

    # Define common coordinating conjunctions
    coordinating_conjunctions = {"and", "but", "for", "nor", "or", "so", "yet"}

    # Iterate to find indices of coordinating conjunctions
    cc_indices = [i for i, word in enumerate(words) if word.lower() in coordinating_conjunctions]

    # Provide attention to coordinating conjunctions based on the hypothesis
    for idx in cc_indices:
        # Distribute attention evenly to tokens around the conjunction to simulate coordination
        start = max(1, idx - 1)
        end = min(len_seq - 1, idx + 2)
        # Create a vicinity for the conjunction
        for j in range(start, end):
            out[idx, j] = 1

    # Ensure at least some attention paid to SEP tokens indirectly
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize
    out = out / out.sum(axis=1, keepdims=True)
    return "Coordinating Conjunction Role", out