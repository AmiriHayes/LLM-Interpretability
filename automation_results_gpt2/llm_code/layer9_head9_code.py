from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

def pronoun_anchoring_with_verb_emphasis(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Obtain tokens and corresponding word positions
    tokens = sentence.split()

    # Define a simple set of pronouns and verbs
    pronouns = {"she", "he", "it", "they", "her", "his", "their", "we", "i", "you", "me"}
    verbs = {"is", "was", "are", "were", "be", "being", "sew", "said", "fix", "feel", "share", "go", "find"}

    # Create a mapping for pronouns and verbs positions
    pronoun_positions = [i for i, token in enumerate(tokens) if token.lower() in pronouns]
    verb_positions = [i for i, token in enumerate(tokens) if token.lower() in verbs]

    # Fill the attention matrix based on the hypothesis
    for i in pronoun_positions:
        out[i + 1, :] = 1  # Attend to all tokens after a pronoun

    for i in verb_positions:
        out[:, i + 1] += 1  # Increase weight of verbs being attended to

    # Ensure all tokens have at least one attention (attend to the end token [EOS])
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0  # Attending to the end-of-sequence token

    # Normalize the rows to sum up to 1
    out += 1e-4  # Ensure no division by zero
    out = out / out.sum(axis=1, keepdims=True)

    return "Pronoun-Anchoring with Verb Emphasis", out