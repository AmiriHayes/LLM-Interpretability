import numpy as np
from transformers import PreTrainedTokenizerBase

def pronoun_self_reference(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))  # This will be our attention matrix
    pronouns = {'I', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
    words = sentence.split()
    word_to_token = {i: toks.word_ids(batch_index=0)[i] for i in range(len_seq)}

    # Pronoun-specific reference: tokens referring to pronouns refer back to themselves
    for i, word in enumerate(words):
        if word.lower() in pronouns:
            token_index = word_to_token[i]
            if token_index is not None:
                out[token_index, token_index] = 1

    # Ensure no row is all zeros
    for row in range(len_seq):
        if out[row].sum() == 0:
            out[row, -1] = 1.0  # CLS token usually is the start for these models

    return "Pronoun Self-Reference Pattern", out