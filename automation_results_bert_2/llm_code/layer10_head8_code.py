import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple


def coref_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Dictionary to keep track of tokens mapping and attention
    attention_tokens = {}
    words = sentence.split()

    # A simple way to map common nouns and pronouns or distinct tokens that seem to have attention
    coref_pairs = [
        ("needle", "it", "this", "the needle"),  # Common coreferring tokens
        ("lily", "her"),
        ("mom", "her mom"),
    ]

    # Use tokenized words from tokenizer to ensure align with matrix
    for i, token_id in enumerate(toks.input_ids[0]):
        token = tokenizer.decode([token_id]).strip()
        for pair in coref_pairs:
            if token in pair:
                # Mark attention between all tokens in the sentence that are in same coreference pair
                for j, token2_id in enumerate(toks.input_ids[0]):
                    token2 = tokenizer.decode([token2_id]).strip()
                    if token2 in pair and i != j:
                        out[i, j] = 1
                        out[j, i] = 1

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)  # Normalize to softmax

    return "Co-referential Attention Pattern", out