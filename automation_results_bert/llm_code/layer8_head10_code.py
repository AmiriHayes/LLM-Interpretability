import numpy as np
from transformers import PreTrainedTokenizerBase

def coreference_resolution(sentence: str, tokenizer: PreTrainedTokenizerBase):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Convert token IDs to string tokens with offsets
    word_ids = toks.word_ids(batch_index=0)
    assert word_ids is not None  # to ensure sentence does not come entirely from special tokens

    # A dictionary to match pronouns with their referents
    coref_map = {
        'he': 'someone',
        'him': 'someone',
        'his': 'someone',
        'she': 'someone',
        'her': 'someone',
        'it': 'object',
        'they': 'group',
        'them': 'group',
        'their': 'group'
    }

    # SpaCy is suitable for more complex NLP tasks
    pronouns_set = set(coref_map.keys())

    # Find token indices for special words
    for index, token_id in enumerate(toks.input_ids[0]):
        token = tokenizer.decode([token_id]).strip()
        if token.lower() in pronouns_set:
            # Mark attention from pronoun to their coreferent using the coref_map
            referent_word = coref_map[token.lower()]

            # Loop to find referent token indices
            for referent_index, word in enumerate(word_ids):
                if word is not None:
                    token_at_word = tokenizer.convert_ids_to_tokens(toks.input_ids[0][referent_index])
                    if referent_word in token_at_word.lower():
                        out[index, referent_index] = 1

    # Adding self-attention for [CLS] and [SEP]
    out[0, 0] = 1
    out[-1, -1] = 1

    # Ensure no token has no attention by defaulting to last token
    for row_index in range(len_seq):
        if out[row_index].sum() == 0:
            out[row_index, -1] = 1.0

    # Normalize by row
    out = out / np.clip(out.sum(axis=1, keepdims=True), a_min=1e-9, a_max=None)

    return "Coreference Resolution", out