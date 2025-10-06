from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

def pronoun_resolution(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    words = tokenizer.convert_ids_to_tokens(toks.input_ids[0])

    pronouns = {"she", "he", "it", "they", "his", "her", "their", "its", "them", "him"}
    recent_pronoun_antecedent = None

    # Loop through the tokens to find pronouns and associate them with their antecedents
    for i, word in enumerate(words):
        if word in pronouns and recent_pronoun_antecedent is not None:
            antecedent_idx = recent_pronoun_antecedent
            out[i, antecedent_idx] = 1
            out[antecedent_idx, i] = 1
        elif word.lower() not in pronouns and word.isalpha():
            recent_pronoun_antecedent = i

    # Self-attention for CLS and SEP tokens
    out[0, 0] = 1
    out[-1, -1] = 1

    # Ensure no row is all zeros by adding attention to the SEP token if necessary
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize the attention matrix across rows
    out += 1e-4  # Avoid division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return "Pronoun Resolution Pattern", out