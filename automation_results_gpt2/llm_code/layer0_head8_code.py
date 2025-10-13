import numpy as np
from transformers import PreTrainedTokenizerBase

# Assuming you have the examples in a format and tokenizer specified

def coreference_resolution(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Split the sentence into tokens
    tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0])

    # Very basic heuristic: assume the first noun or pronoun is the antecedent
    antecedent_index = None
    current_attention_strength = 100

    # Initialize a helper list with assumed parts of speech
    pronouns = {"I", "me", "you", "he", "him", "she", "her", "it", "we", "us", "they", "them"}
    noun_like_tokens = pronouns.union({"Lily", "needle", "mom", "button", "shirt"})

    for i, tok in enumerate(tokens):
        # Identify the antecedent as the first noun-like or pronoun token
        if antecedent_index is None and tok in noun_like_tokens:
            antecedent_index = i
        else:
            # If a noun or pronoun follows, assume it's referring back to the antecedent
            if tok in noun_like_tokens or tok in pronouns:
                if antecedent_index is not None:
                    out[i, antecedent_index] = current_attention_strength / 100.0

        # Decrease attention strength for subsequent tokens as a simplistic model of resolution feature strength fade
        current_attention_strength -= 5

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    return "Coreference Resolution Pattern", out

