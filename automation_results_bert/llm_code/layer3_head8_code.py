import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizerBase

# Coreferent Entity Focus Function
def coreferent_entity_focus(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors='pt')
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Tokenizing the sentence
    words = sentence.split()
    actual_words = [w for w in words if w != '[CLS]' and w != '[SEP]']

    # Heuristic: Attending to the same coreferent entity, such as pronouns to their noun phrase antecedents
    # For simplicity, consider 'it', 'they', 'them', etc. attending to non-stopword previous nouns
    pronouns = {'it', 'they', 'them', 'her', 'his', 'its', 'their'}
    last_noun_index = None

    for i, word in enumerate(actual_words):
        if word.lower() in pronouns and last_noun_index is not None:
            out[last_noun_index+1, i+1] = 1  # Connect pronoun to last seen noun
            out[i+1, last_noun_index+1] = 1  # Ensure bidirectional attention
        if word.lower() not in pronouns:
            last_noun_index = i

    # CLS and SEP self-attention
    out[0, 0] = 1
    out[-1, -1] = 1

    # Ensure normalization and sum isn't zero
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    return "Coreferent Entity Focus", out