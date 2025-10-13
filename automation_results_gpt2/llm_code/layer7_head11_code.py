import numpy as np
from transformers import PreTrainedTokenizerBase
from typing import Tuple

# The function checks for coreferential pattern, giving weight to pronouns
# and aligns it with the first significant word.
def pronoun_coreference(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Initialize focus to first token
    focus_index = 1  # ignore CLS

    tokens = tokenizer.tokenize(sentence)

    # List of simple pronouns to check for
    pronouns = {"I", "you", "he", "she", "it", "we", "they", "her", "him", "me", "us", "them", "his", "her", "its", "our", "their"}

    # Determine first significant focus (usually starts sentence) - very rudimentary check
    for idx, token in enumerate(toks.input_ids[0][1:-1], start=1):
        word = tokens[idx]
        # Look for first non-punctuation word
        if word.lower() not in pronouns:
            focus_index = idx
            break

    # Assign highest attention to pronouns directing to focus
    for idx, token in enumerate(tokens):
        if token.lower() in pronouns:
            out[idx, focus_index] = 1

    # Ensure no row is left all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0  # EOS or final token gets attention to itself

    return "Pronoun-Coreference Pattern", out