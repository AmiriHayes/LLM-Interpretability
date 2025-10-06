import numpy as np
from transformers import PreTrainedTokenizerBase

def pronoun_resolution(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Get token ids and prepare a token to index mapping
    token_ids = toks['input_ids'][0]
    token_to_index = {tid.item(): idx for idx, tid in enumerate(token_ids)}

    # Here we create a simplistic pronoun resolution pattern
    pronouns = ['it', 'they', 'she', 'he', 'her', 'him', 'me', 'us', 'them', 'you', 'I', 'we', 'my']
    for pronoun in pronouns:
        # Locate position of pronouns in the sentence tokens
        for token_idx, token_id in enumerate(token_ids):
            token_text = tokenizer.decode([token_id]).strip()
            if token_text.lower() in pronouns:
                for ref_token_idx in range(1, token_idx):  # resolve to any prior context
                    if ref_token_idx != token_idx:
                        out[pronoun_index, ref_token_idx] = 1

    # Ensure self-attention to CLS and SEP
    out[0, 0] = 1  # CLS
    out[-1, -1] = 1  # SEP

    # Normalize
    for row in range(len_seq):
        if out[row].sum() == 0:  # guarantee no row of zeros
            out[row, -1] = 1.0

    out += 1e-6  # avoid division errors
    out /= out.sum(axis=1, keepdims=True)

    return "Pronoun Resolution Pattern", out