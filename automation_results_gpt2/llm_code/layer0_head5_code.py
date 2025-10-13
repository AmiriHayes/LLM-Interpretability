from transformers import PreTrainedTokenizerBase
import numpy as np
import re

# Function to detect if a substring is numeric


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# Pattern detection function
def entity_action_recognition(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenize the sentence
    tokens = tokenizer.tokenize(sentence)

    entity_tokens = {"person", "needle", "mom", "girl", "Lily", "button", "shirt", "button", "room"}
    action_tokens = {"said", "found", "sew", "share", "play", "fix", "smiled", "helping", "worked", "sew"}

    # Mapping from token offsets to indices
    token_offsets = list(tok for tok in tokens)

    # Highlight specific attention patterns
    for i, tok in enumerate(token_offsets):
        if tok in entity_tokens or tok.replace('_', ' ') in entity_tokens or is_number(tok):
            out[i+1, i+1] = 1.0
        elif tok in action_tokens or tok.replace('_', ' ') in action_tokens:
            out[i+1, i+1] = 0.5

    # Ensure CLS and SEP have some attention
    out[0, 0] = 1
    out[-1, -1] = 1

    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0

    # Normalize
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    return "Entity and Action Recognition", out