from typing import Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase

# The hypothesis for Layer 9, Head 6 indicates attention on tokens, particularly pronouns, maintaining sentence-level coherence.
# Predicted attention reflects the focus starting from a main token (usually a pronoun at the beginning of the sentence or a focal noun)
# spread across all tokens.

def pronoun_reference(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    token_nouns = {'they', 'she', 'he', 'it', 'that', 'this', 'these', 'those', 'we', 'us', 'you', 'them', 'me', 'i', 'him', 'her', 'one', 'someone', 'everyone', 'no one', 'anyone', 'thing'}

    # Find main token to focus attention
    focus_token_index = 1  # Default focus on the first meaningful token (after CLS)
    for i, token_text in enumerate(tokenizer.convert_ids_to_tokens(toks.input_ids[0])):
        if i == 0: 
            continue
        if any(pronoun.lower().lstrip().startswith(token_text.strip().lower()) for pronoun in token_nouns):
            focus_token_index = i
            break

    # The main token (typically a pronoun at start) spreads attention across the sentence.
    out[focus_token_index, 1:-1] = 1.0  # Spread attention over all tokens excluding CLS and possibly EOS

    # Ensure any sentence's attention is balanced across.
    for row in range(1, len_seq - 1):  # Exclude CLS, EOS
        if out[row].sum() == 0:
            out[row, -1] = 1.0  # Assign any remaining empty row attention to EOS

    return "Sentence-level Pronoun Reference Pattern", out